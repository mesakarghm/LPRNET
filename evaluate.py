import tensorflow as tf
import editdistance
import numpy as np
class Evaluator: 

    def __init__(self,val_gen, net, class_names,val_batch_len,batch_size):
        self.net = net 
        self.val_gen = val_gen 
        self.class_names = class_names
        self.batch_size = batch_size
        self.val_gen = val_gen
        self.val_batch_len = val_batch_len

    def _average(self, values):
        if len(values) == 1:
            return values[0]
        return(np.sum(values )/self.val_batch_len)
        # return (np.sum(values[:-1] * self.batch_size) + values[-1] * last_batch_size) / len(self.loader)


    def _decode_label(self,labels):
        results = []
        for d in labels: 
            text = []
            for idx in d: 
                if idx == -1: 
                    break 
                text.append(self.class_names[idx])
            results.append(''.join(text).encode('utf-8'))
        return results 

    def decode_pred(self,pred ):
        samples, times = pred.shape[:2]
        input_length = tf.convert_to_tensor([times] * samples)
        decodeds, logprobs = tf.keras.backend.ctc_decode(pred, input_length, greedy=True, beam_width=100, top_paths=1)
        decodeds = np.array(decodeds[0])

        results = []
        for d in decodeds:
            text = []
            for idx in d:
                if idx == -1:
                    break
                text.append(self.class_names[idx])
            results.append(''.join(text).encode('utf-8'))
        return results

    def _calc_CER_and_WER(self, label_texts, decoded_texts):
        ed = []
        WER = 0
        for label, pred in zip(label_texts, decoded_texts):
            print("label \t {} \t prediction \t {}".format(label,pred))
            cer = editdistance.eval(label, pred)
            ed.append(cer)
            if cer != 0:
                WER += 1
        WER /= len(label_texts)
        CER = sum(ed) / len(label_texts)
        return CER, WER

    def _print_result(self, loss, CER, WER):
        print("Number of samples in test set: {}\n"
              "mean loss: {}\n"
              "mean CER: {}\n"
              "WER: {}\n".format(self.val_batch_len * self.batch_size,
                                 loss,
                                 CER,
                                 WER
                                 )
              )

    def evaluate(self): 
        self.losses, self.CERs, self.WERs = [],[],[]

        for val_batch in range(self.val_batch_len):
            val_inputs, val_targets, val_labels = self.val_gen.next_batch()
            val_inputs = val_inputs.astype('float32')
            val_targets = tf.SparseTensor(val_targets[0], val_targets[1], val_targets[2])
            logits = self.net.model(val_inputs, training = False)
            logits = tf.reduce_mean(logits, axis = 1)
            decoded_texts = self.decode_pred(logits)
            label_texts = self._decode_label(val_labels)
            CER,WER = self._calc_CER_and_WER(label_texts, decoded_texts)

            logits_shape = tf.shape(logits)
            seq_len = tf.fill([logits_shape[0]],logits_shape[1])
            logits = tf.transpose(logits, (1,0,2))
            loss_value = tf.reduce_mean(tf.nn.ctc_loss(labels = val_targets, inputs = logits, sequence_length = seq_len ))
            # print("Loss: {} - CER: {}, WER:{}\n".format(float(loss_value),CER,WER))
            self.losses.append(float(loss_value))
            self.CERs.append(CER)
            self.WERs.append(WER)
        loss = self._average(self.losses)
        cer = self._average(self.CERs)
        wer = self._average(self.WERs)
        self._print_result(loss,cer,wer)
        return(loss)