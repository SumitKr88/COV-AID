# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
import argparse

class COVAIDApp:

# Parse all arguments
	parser = argparse.ArgumentParser(
		description='Thermal screening demo by Codevector Labs.')
	parser.add_argument('-t', '--threshold_temperature', dest='threshold_temperature', default=100.5, type=float,
						help='Threshold temperature in Farenheit (float)', required=False)
	parser.add_argument('-b', '--binary_threshold', dest='binary_threshold', default=200, type=int,
						help='Threshold pixel value for binary threshold (between 0-255)', required=False)
	parser.add_argument('-c', '--conversion_factor', dest='conversion_factor', default=2.25, type=float,
						help='Conversion factor to convert pixel value to temperature (float)', required=False)
	parser.add_argument('-a', '--min_area', dest='min_area', default=2400, type=int,
						help='Minimum area of the rectangle to consider for further porcessing (int)', required=False)
	parser.add_argument('-i', '--input_video', dest='input_video', default=os.path.join("data", "input.mp4"), type=str,
						help='Input video file path (string)', required=False)
	parser.add_argument('-o', '--output_video', dest='output_video', default=os.path.join("output", "output.avi"), type=str,
						help='Output video file path (string)', required=False)
	parser.add_argument('-f', '--fps', dest='fps', default=15, type=int,
						help='FPS of output video (int)', required=False)
	args = parser.parse_args().__dict__


	# Global Variables
	threshold_temperature = 90.0
	currentUnit = "f"

	def convert_to_temperature(self, pixel_avg):
		"""
		Converts pixel value (mean) to temperature (farenheit) depending upon the camera hardware
		"""
		return pixel_avg / self.args['conversion_factor']

	def process_temperature_frame(self, frame):
		
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		heatmap_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)
		image_with_rectangles = np.copy(heatmap)

		# Binary threshold
		_, binary_thresh = cv2.threshold(
			heatmap_gray, self.args['binary_threshold'], 255, cv2.THRESH_BINARY)

		# Image opening: Erosion followed by dilation
		kernel = np.ones((5, 5), np.uint8)
		image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
		image_opening = cv2.dilate(image_erosion, kernel, iterations=1)

		# Get contours from the image obtained by opening operation
		contours, _ = cv2.findContours(image_opening, 1, 2)
		temperatureArray = [0.0]

		for contour in contours:
			# rectangle over each contour
			x, y, w, h = cv2.boundingRect(contour)

			# Pass if the area of rectangle is not large enough
			if (w) * (h) < self.args['min_area']:
				continue

			# Mask is boolean type of matrix.
			mask = np.zeros_like(heatmap_gray)
			cv2.drawContours(mask, contour, -1, 255, -1)

			# Mean of only those pixels which are in blocks and not the whole rectangle selected
			mean = self.convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])

			# Colors for rectangles and textmin_area
			temperature = round(mean, 2)
			if temperature:
				temperatureArray.append(temperature)
		return max(temperatureArray)



	def process_temperature_face_frame(self, frame):
		# grab the dimensions of the frame and then construct a blob
		# from it
		image_with_rectangles = np.copy(frame)
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=True)

		# pass the blob through the network and obtain the face detections
		self.faceNet.setInput(blob)
		detections = self.faceNet.forward()
		print(detections.shape)
		print(self.threshold_temperature)

		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		locs = []
		preds = []

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = frame[startY:endY, startX:endX]
				maxTemperature = self.process_temperature_frame(frame)

				color = (255, 0, 0)

				# Draw rectangles for visualisation
				image_with_rectangles = cv2.rectangle(
					image_with_rectangles, (startX, startY), (endX, endY), color, 2)

				# Write temperature for each rectangle
				cv2.putText(image_with_rectangles, "{} F".format(maxTemperature), (startX, startY),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
		return image_with_rectangles


	def detect_and_predict_mask(self, frame, faceNet, maskNet):
		# grab the dimensions of the frame and then construct a blob
		# from it
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=True)

		# pass the blob through the network and obtain the face detections
		faceNet.setInput(blob)
		detections = faceNet.forward()
		# print(detections.shape)

		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		locs = []
		preds = []
		faceTemperatureList = []

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = frame[startY:endY, startX:endX]
				maxTemperature = self.process_temperature_frame(face)
				faceTemperatureList.append(maxTemperature)
				# face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = tf.keras.applications.mobilenet.preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

		# only make a predictions if at least one face was detected
		if len(faces) > 0:
			# for faster inference we'll make batch predictions on *all*
			# faces at the same time rather than one-by-one predictions
			# in the above `for` loop
			faces = np.array(faces, dtype="float32")
			preds = maskNet.predict(faces, batch_size=32)

		# return a 2-tuple of the face locations and their corresponding
		# locations
		return (locs, preds, faceTemperatureList)


	def __init__(self, vs):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.vs = vs
		self.frame = None
		self.thread = None
		self.stopEvent = None

		prototxtPath = r"face_detector/deploy.prototxt"
		weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
		self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

		# load the face mask detector model from disk
		self.maskNet = load_model("face_mask_model.h5")

		# initialize the root window and image panel
		self.root = tki.Tk()
		self.isFahrenheit = tki.IntVar()

		self.panel = None
		bottomframe = tki.Frame(self.root)
		bottomframe.pack(side="bottom")

		# create a button, that when pressed, will take the current
		# frame and save it to file
		# label = tki.Label(self, text ="Temp Treshold")
		# label.pack(side="bottom", fill="both", expand="yes", padx=10,
			# pady=10)
		middleframe = tki.Frame(bottomframe)
		middleframe.pack(side="top")

		temperatureframe = tki.Frame(middleframe)
		temperatureframe.pack(side="top")
		label = tki.Label(temperatureframe, text="Temperature Threshold")
		label.pack(side="left", fill="both", expand="yes", padx=10,
			pady=10)

		self.textBox = tki.Entry(temperatureframe)
		self.textBox.pack(side="right", fill="both", expand="yes", padx=10,
			pady=10)

		unitsframe = tki.Frame(middleframe)
		unitsframe.pack(side="bottom")
		label = tki.Label(unitsframe, text="Units")
		label.pack(side="left", fill="both", expand="yes", padx=10,
			pady=10)

		R1 = tki.Radiobutton(unitsframe, text="Celcius", variable=self.isFahrenheit, value=1,  command=self.sel)
		R1.pack(side="left", fill="both", expand="yes", padx=10,
			pady=10)

		R2 = tki.Radiobutton(unitsframe, text="Farenheit", variable=self.isFahrenheit, value=0,  command=self.sel)
		R2.pack(side="left", fill="both", expand="yes", padx=10,
			pady=10)

		
		legendFrame = tki.Frame(bottomframe)
		legendFrame.pack(side="bottom")
		btn = tki.Button(legendFrame, text="Submit",
		command=self.takeSnapshot)
		btn.pack(side="top", fill="both", expand="yes", padx=10,
			pady=10)
		redLabel = tki.Label(legendFrame, text="A Red coloured box indicates your temperature is higher than the Threshold", anchor='w')
		redLabel.pack(side="top", fill="both", expand="yes", padx=10,
			pady=10)
		yellowLabel = tki.Label(legendFrame, text="A Yellow coloured box indicates your temperature is within Threshold, but your mask is not detected", anchor='w')
		yellowLabel.pack(side="top", fill="both", expand="yes", padx=10,
			pady=10)
		greenLabel = tki.Label(legendFrame, text="A Green coloured box indicates you satisfy the given safety protocol", anchor='w')
		greenLabel.pack(side="top", fill="both", expand="yes", padx=10,
			pady=10)

		
		# start a thread that constantly pools the video sensor for
		# the most recently read frame
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()

		# set a callback to handle when the window is closed
		self.root.wm_title("COV-AID")
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

	def sel(self):
		print(self.isFahrenheit.get())
		if self.currentUnit == "f" and self.isFahrenheit.get() == 1:
			self.currentUnit = "c"
			self.threshold_temperature = 5 * (self.threshold_temperature - 32) / 9
		elif self.currentUnit == "c" and self.isFahrenheit.get() == 0:
			self.currentUnit = "f"
			self.threshold_temperature = (9 * self.threshold_temperature + 160) / 5




	def videoLoop(self):
		# DISCLAIMER:
		# I'm not a GUI developer, nor do I even pretend to be. This
		# try/except statement is a pretty ugly hack to get around
		# a RunTime error that Tkinter throws due to threading
		try:
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				self.frame = self.vs.read()
				self.frame = imutils.resize(self.frame, width=1300)
				# detect faces in the frame and determine if they are wearing a
				# face mask or not
				(locs, preds, faceTemperatureList) = self.detect_and_predict_mask(self.frame, self.faceNet, self.maskNet)

				# loop over the detected face locations and their corresponding
				# locations
				for (box, pred, temperature) in zip(locs, preds, faceTemperatureList):
					# unpack the bounding box and predictions
					(startX, startY, endX, endY) = box
					(mask, withoutMask) = pred

					# determine the class label and color we'll use to draw
					# the bounding box and text
					isHighTemperature = False
					isNoMask = False
					if mask > withoutMask:
						maskLabel = "Mask" 
					else:
						maskLabel = "No Mask"
						isNoMask = True
					
					# include the probability in the label
					maskLabel = "{}: {:.2f}%".format(maskLabel, max(mask, withoutMask) * 100)
					temperatureLabel = ""
					# print(self.threshold_temperature)
					if self.isFahrenheit.get() == 1:
						temperature = 5 * (temperature - 32) / 9
					if temperature < self.threshold_temperature:
						temperatureLabel = "Temperature: " + str(temperature) + ("F" if self.isFahrenheit.get() == 0 else "C")
					else:
						temperatureLabel = "High temperature: " + str(temperature) + ("F" if self.isFahrenheit.get() == 0 else "C")
						isHighTemperature = True

					temperatureColor = (0, 93, 196)
					boxColor = (0, 255, 0)
					if (isHighTemperature and isNoMask) or (isHighTemperature):
						boxColor = (0, 0, 255)
					elif isNoMask:
						boxColor = (0, 255, 255)

					# display the label and bounding box rectangle on the output
					# frame
					cv2.putText(self.frame, maskLabel, (startX, startY - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, boxColor, 2)
					cv2.putText(self.frame, temperatureLabel, (startX, endY + 20),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, temperatureColor, 2)
					cv2.rectangle(self.frame, (startX, startY), (endX, endY), boxColor, 2)
		
				# if the panel is not None, we need to initialize it
				# OpenCV represents images in BGR order; however PIL
				# represents images in RGB order, so we need to swap
				# the channels, then convert to PIL and ImageTk format
				image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(image)
				image = ImageTk.PhotoImage(image)

				if self.panel is None:
					self.panel = tki.Label(image=image)
					self.panel.image = image
					self.panel.pack(side="left", padx=10, pady=10)

				# otherwise, simply update the panel
				else:
					self.panel.configure(image=image)
					self.panel.image = image

		except RuntimeError as e:
			print("[INFO] caught a RuntimeError")


	
	def takeSnapshot(self):
		# grab the current timestamp and use it to construct the
		# output path
		self.threshold_temperature = float(self.textBox.get())

	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		cv2.destroyAllWindows()
		self.stopEvent.set()
		self.vs.stop()
		self.root.quit()

