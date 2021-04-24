import argparse 
from time import time 

import numpy as np 
import cv2 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 

from model import LPRNet 
from tensorflow.keras.models import load_model
from model import LPRNet


# from layers import BilinearInterpolation
import sys 
sys.path.insert(1,"./py")
import codecs 
# from ./py/WordBeamSearch import wordBeamSearch
# from LanguageModel import LanguageModel
# from WordBeamSearch import wordBeamSearch
from word_beam_search import WordBeamSearch
classnames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


IMG_SIZE = [94,24]

def run():
	model = load_model("./saved_models/50r_50g_epoch_400.pb")
	print("Loaded Weights successfully")
	print("Actual Label \t Predicted Label ")
	start_time = time()
	cnt = 0
	for filename in os.listdir("./samples"): 
		if filename.endswith(".jpg") or filename.endswith(".JPG"): 
			frame = cv2.imread(f"./samples/{filename}")
			img = cv2.resize(frame, (94,24))
			img = np.expand_dims(img,axis = 0)
			pred = model.predict(img)
			result_ctc = decode_pred(pred, classnames)
			original_label = filename.split("_")[0]

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


def softmax(mat):
		"calc softmax such that labels per time-step form probability distribution"
		maxT, _ = mat.shape[:2]
		res = np.zeros(mat.shape)
		for t in range(maxT):
			y = mat[t, :]
			e = np.exp(y)
			s = np.sum(e)
			res[t, :] = e/s
		return res

if __name__ == "__main__": 
	print("Starting program...")
	tf.compat.v1.enable_eager_execution() 
	run()