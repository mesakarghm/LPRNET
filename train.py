import tensorflow as tf 
from tensorflow import keras
import os 
import time 
import argparse
import utils 
import math
from model import LPRNet
import evaluate

import numpy as np 
from tensorflow.keras import backend as K 

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NUM_CLASS = len(CHARS)+1

def train(checkpoint,train_dir = "./train", val_dir="valid",batch_size = 8,val_batch_size = 1, train_epochs = 10000):
	net = LPRNet(NUM_CLASS)   #(use for KoreanLPR based model)
	train_gen = utils.DataIterator(img_dir=train_dir, batch_size = batch_size)
	val_gen = utils.DataIterator(img_dir=val_dir,batch_size = val_batch_size)
	train_len = len(next(os.walk(train_dir))[2])
	val_len = len(next(os.walk(val_dir))[2])
	print("Train Len is", train_len)
	BATCH_PER_EPOCH = None 
	if batch_size ==1: 
		BATCH_PER_EPOCH = train_len 
	else: 
		BATCH_PER_EPOCH = int(math.ceil(train_len/batch_size))

	val_batch_len = int(math.floor(val_len / val_batch_size))  
	evaluator = evaluate.Evaluator(val_gen,net, CHARS,val_batch_len, val_batch_size)
	best_val_loss = float("inf")
	if args["pretrained"]:
		net.load_weights(args["pretrained"])

	model = net.model
	learning_rate = keras.optimizers.schedules.ExponentialDecay(args["lr"],
															decay_steps=args["decay_steps"],
															decay_rate=args["decay_rate"],
															staircase=args["staircase"])

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	print('Training ...')
	train_loss = 0 

	for epoch in range(train_epochs):
		
		print("Start of epoch {} / {}".format(epoch,train_epochs))
		train_loss = 0 
		val_loss = 0
		start_time = time.time() 
		for batch in range(BATCH_PER_EPOCH):
			# print("batch {}/{}".format(batch, BATCH_PER_EPOCH))
			train_inputs,train_targets,train_labels = train_gen.next_batch()
			train_inputs = train_inputs.astype('float32')

			train_targets = tf.SparseTensor(train_targets[0],train_targets[1],train_targets[2])


		# Open a GradientTape to record the operations run
		# during the forward pass, which enables auto-differentiation.
			with tf.GradientTape() as tape: 

				logits = model(train_inputs,training = True)
				logits = tf.reduce_mean(logits, axis = 1)
				logits_shape = tf.shape(logits)
				cur_batch_size = logits_shape[0]
				timesteps = logits_shape[1]
				seq_len = tf.fill([cur_batch_size],timesteps)
				logits = tf.transpose(logits,(1,0,2))
				ctc_loss = tf.nn.ctc_loss(labels = train_targets, inputs = logits, sequence_length = seq_len)
				loss_value =tf.reduce_mean(ctc_loss)

  
			#Calculate Gradients and Update it 
			grads = tape.gradient(ctc_loss, model.trainable_weights,unconnected_gradients=tf.UnconnectedGradients.NONE)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
			train_loss += float(loss_value)

		tim = time.time() - start_time

		print("Train loss {}, time {} \n".format(float(train_loss/BATCH_PER_EPOCH),tim))
		if epoch != 0 and epoch%10 == 0:
			evaluator.evaluate()
			net.save_weights(os.path.join(args["saved_dir"], "new_out_model_best.pb"))
			print("Weights updated in {}/{}".format(args["saved_dir"],"new_out_model_best.pb"))

			if epoch %100 == 0: 
				net.save(os.path.join(args["saved_dir"], f"new_out_model_last_{epoch}.pb"))




	net.save(os.path.join(args["saved_dir"], "new_out_model_last.pb"))
	print("Final Weights saved in {}/{}".format(args["saved_dir"], "new_out_model_last.pb"))



def parser_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--train_dir", help = "path to the train directory")
	parser.add_argument("--val_dir", help = "path to the validation directory")

	parser.add_argument("--train_epchs", type = int, help = "number of training epochs")
	parser.add_argument("--batch_size", type = int, help = "batch size (train)")
	parser.add_argument("--val_batch_size", type = int, help = "Validation batch size")
	parser.add_argument("--lr", type = float, default = 1e-3, help = "initial learning rate")
	parser.add_argument("--decay_steps", type = float, default = 1000, help = "learning rate decay rate")
	parser.add_argument("--decay_rate", type=float, default=0.995, help="learning rate decay rate")
	parser.add_argument("--staircase", action = "store_true", help = "learning rate decay on step (default:smooth)")

	parser.add_argument("--pretrained", help = "pretrained model location ")
	parser.add_argument("--saved_dir", default = "saved_models", help = "folder for saving models")

	args = vars(parser.parse_args())
	return args

if __name__ == "__main__":
	args = parser_args()

	tf.compat.v1.enable_eager_execution()
	train(args)

