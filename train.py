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

##implementing our custom training loop 
##This supports variable training labels which is the case in some License Plates across the countries 


def train():

	#Initiate the Neural Network 
	net = LPRNet(NUM_CLASS) 

	#get the trainn and validation batch size from argument parser
	batch_size = args["batch_size"]
	val_batch_size = args["val_batch_size"]

	#initialize the custom data generator 
	#Defined in utils.py
	train_gen = utils.DataIterator(img_dir=args["train_dir"], batch_size = batch_size)
	val_gen = utils.DataIterator(img_dir=args["val_dir"],batch_size = val_batch_size)

	#variable intialization used for custom training loop 
	train_len = len(next(os.walk(args["train_dir"]))[2])
	val_len = len(next(os.walk(args["val_dir"]))[2])
	print("Train Len is", train_len)
	# BATCH_PER_EPOCH = None 
	if batch_size ==1: 
		BATCH_PER_EPOCH = train_len 
	else: 
		BATCH_PER_EPOCH = int(math.ceil(train_len/batch_size))

	val_batch_len = int(math.floor(val_len / val_batch_size))  
	evaluator = evaluate.Evaluator(val_gen,net, CHARS,val_batch_len, val_batch_size)
	best_val_loss = float("inf")

	#if a pretrained model is available, load weights from it 
	if args["pretrained"]:
		net.load_weights(args["pretrained"])


	model = net.model

	#initialize the learning rate
	learning_rate = keras.optimizers.schedules.ExponentialDecay(args["lr"],
															decay_steps=args["decay_steps"],
															decay_rate=args["decay_rate"],
															staircase=args["staircase"])

	#define training optimizer 
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	print('Training ...')
	train_loss = 0 

	#starting the training loop 
	for epoch in range(args["train_epochs"]):
		
		print("Start of epoch {} / {}".format(epoch,args["train_epochs"]))

		#zero out the train_loss and val_loss at the beginning of every loop 
		#This helps us track the loss value for every epoch 
		train_loss = 0 
		val_loss = 0
		start_time = time.time() 

		for batch in range(BATCH_PER_EPOCH):
			# print("batch {}/{}".format(batch, BATCH_PER_EPOCH))
			#get a batch of images/labels
			#the labels have to be put into sparse tensor to feed into tf.nn.ctc_loss 
			train_inputs,train_targets,train_labels = train_gen.next_batch()
			train_inputs = train_inputs.astype('float32')

			train_targets = tf.SparseTensor(train_targets[0],train_targets[1],train_targets[2])


		# Open a GradientTape to record the operations run
		# during the forward pass, which enables auto-differentiation.
			with tf.GradientTape() as tape: 

				#get model outputs
				logits = model(train_inputs,training = True)

				#next we pass the model outputs into the ctc loss function
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

		#run a validation loop in every 25 epoch
		if epoch != 0 and epoch%25 == 0:
			val_loss = evaluator.evaluate()
			#if the validation loss is less the previous best validation loss, update the saved model
			if val_loss < best_val_loss: 
				best_val_loss = val_loss
				net.save_weights(os.path.join(args["saved_dir"], "new_out_model_best.pb"))
				print("Weights updated in {}/{}".format(args["saved_dir"],"new_out_model_best.pb"))

			else: 
				print("Validation loss is greater than best_val_loss ")

			if epoch %100 == 0: 
				net.save(os.path.join(args["saved_dir"], f"new_out_model_last_{epoch}.pb"))




	net.save(os.path.join(args["saved_dir"], "new_out_model_last.pb"))
	print("Final Weights saved in {}/{}".format(args["saved_dir"], "new_out_model_last.pb"))



def parser_args():
	"""
	Argument Parser for command line arguments
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument("--train_dir",default = "./car_video", help = "path to the train directory",)
	parser.add_argument("--val_dir",default = "./car_video", help = "path to the validation directory")

	parser.add_argument("--train_epochs", type = int, help = "number of training epochs", default = 1000)
	parser.add_argument("--batch_size", type = int,default = 8, help = "batch size (train)")
	parser.add_argument("--val_batch_size", type = int,default = 4, help = "Validation batch size")
	parser.add_argument("--lr", type = float, default = 1e-3, help = "initial learning rate")
	parser.add_argument("--decay_steps", type = float, default = 500, help = "learning rate decay rate")
	parser.add_argument("--decay_rate", type=float, default=0.995, help="learning rate decay rate")
	parser.add_argument("--staircase", action = "store_true", help = "learning rate decay on step (default:smooth)")

	parser.add_argument("--pretrained", help = "pretrained model location ")
	parser.add_argument("--saved_dir", default = "saved_models", help = "folder for saving models")

	args = vars(parser.parse_args())
	return args

if __name__ == "__main__":
	args = parser_args()
	CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	NUM_CLASS = len(CHARS)+1
	tf.compat.v1.enable_eager_execution()
	train()

