# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import tensorflow as tf
import numpy as np
import imutils
import time
import cv2
import os
import argparse

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

def convert_to_temperature(pixel_avg):
    """
    Converts pixel value (mean) to temperature (farenheit) depending upon the camera hardware
    """
    return pixel_avg / args['conversion_factor']


def process_temperature_face_frame(frame, faceNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    image_with_rectangles = np.copy(frame)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=True)

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

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
            maxTemperature = process_temperature_frame(frame)

            color = (255, 0, 0)

            # Draw rectangles for visualisation
            image_with_rectangles = cv2.rectangle(
                image_with_rectangles, (startX, startY), (endX, endY), color, 2)

            # Write temperature for each rectangle
            cv2.putText(image_with_rectangles, "{} F".format(maxTemperature), (startX, startY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return image_with_rectangles


def process_temperature_frame(frame):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    heatmap_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)
    image_with_rectangles = np.copy(heatmap)

    # Binary threshold
    _, binary_thresh = cv2.threshold(
        heatmap_gray, args['binary_threshold'], 255, cv2.THRESH_BINARY)

    # Image opening: Erosion followed by dilation
    kernel = np.ones((5, 5), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    image_opening = cv2.dilate(image_erosion, kernel, iterations=1)

    # Get contours from the image obtained by opening operation
    contours, _ = cv2.findContours(image_opening, 1, 2)
    temperatureArray = []

    for contour in contours:
        # rectangle over each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Pass if the area of rectangle is not large enough
        if (w) * (h) < args['min_area']:
            continue

        # Mask is boolean type of matrix.
        mask = np.zeros_like(heatmap_gray)
        cv2.drawContours(mask, contour, -1, 255, -1)

        # Mean of only those pixels which are in blocks and not the whole rectangle selected
        mean = convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])

        # Colors for rectangles and textmin_area
        temperature = round(mean, 2)
        temperatureArray.append(temperature)
        
    return max(temperatureArray)





def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=True)

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

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
    return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("face_mask_model.h5")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1080)
    frame = process_temperature_face_frame(frame, faceNet)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
