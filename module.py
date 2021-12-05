import numpy as np
import cv2, os, random, colorsys, uuid, logging
print(cv2.__version__)




class AgeGender(object):
	def __init__(self, path_age, path_gender):
		self.ageNet = cv2.dnn.readNet(path_age)
		self.genderNet = cv2.dnn.readNet(path_gender)
		self.scale = 0.00392156862745098
		self.layer1 = ['StatefulPartitionedCall/StatefulPartitionedCall/model_4/acc_sex/Sigmoid',\
		     		'StatefulPartitionedCall/StatefulPartitionedCall/model_4/acc_age/MatMul']
		self.layer2 = ['StatefulPartitionedCall/StatefulPartitionedCall/model_2/acc_sex/Sigmoid',\
		    		'StatefulPartitionedCall/StatefulPartitionedCall/model_2/acc_age/MatMul']
		self.cls_  = {0:"M",1:"F"}

	def predict(self, img):
		image = cv2.imread(img)
		img_blob_age    = cv2.dnn.blobFromImage(image,   self.scale, (64, 64),   swapRB=True, crop=False)
		img_blob_gender = cv2.dnn.blobFromImage(image,   self.scale, (128, 128), swapRB=True, crop=False)
		self.ageNet.setInput(img_blob_age)
		self.genderNet.setInput(img_blob_gender)
		out1  = self.ageNet.forward(self.layer1)
		out2 = self.genderNet.forward(self.layer2)
		result = {"age":int(np.round(out1[1][0])), 'sex':int(np.round(out2[0][0]))}
		return result['age'], self.cls_[result['sex']]


if __name__ == '__main__':
	logging.basicConfig(filename=f'log/app.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')
	cls = AgeGender(path_age = "weights/frozen_graph_age.pb", path_gender = "weights/frozen_graph_sex.pb")
	logging.info('load model.')
	result = cls.predict('img/8.jpg')
	logging.info(f'[INFO]: prediction -> {result}')











