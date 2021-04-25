import argparse 
from time import time 

import numpy as np 
import cv2 
import os 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 

from model import LPRNet 
from tensorflow.keras.models import load_model

def run():
	#load the KERAS model
	model = load_model("./saved_models/new_out_model_last.pb")
	print("Loaded Weights successfully")
	print("Actual Label \t Predicted Label ")
	start_time = time()
	cnt = 0

	#loop through all the files in the test folder
	for filename in os.listdir("./images/samples"): 
		#check if the file is an image
		if filename.endswith(".jpg") or filename.endswith(".JPG"): 
			#read the file and preprocess it 
			frame = cv2.imread(f"./images/samples/{filename}")
			img = cv2.resize(frame, (94,24))
			img = np.expand_dims(img,axis = 0)
			#get the output sequence
			pred = model.predict(img)

			#decode the output sequence using keras ctc decode 
			result_ctc = decode_pred(pred, classnames)
			original_label = filename.split("_")[0]

			#print the original sequence and the decoded sequence
			print(original_label,"\t",result_ctc[0].decode('utf-8'))
			cnt+=1
	print("total time taken :", time()-start_time)

def decode_pred(pred,classnames):
	pred = np.mean(pred, axis = 1)
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
			text.append(classnames[idx])
		results.append(''.join(text).encode('utf-8'))
	return results



if __name__ == "__main__": 
	print("Starting program...")
	classnames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	IMG_SIZE = [94,24]
	tf.compat.v1.enable_eager_execution() 
	run()