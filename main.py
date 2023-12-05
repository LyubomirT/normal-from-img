from flask import Flask, render_template, request, flash
import cv2
import numpy as np
import os
import base64

import uuid  # For generating unique filenames

import base64
import cv2
import numpy as np

import cv2
import numpy as np
import base64

def normal_map_from_base64(image_base64):
  # decode the image base64 to a numpy array
  image_data = base64.b64decode(image_base64)
  image_array = np.frombuffer(image_data, dtype=np.uint8)
  image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

  # convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # apply a sobel filter to get the x and y gradients
  sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
  sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

  # normalize the gradients to the range [0, 255]
  sobel_x = cv2.normalize(sobel_x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  sobel_y = cv2.normalize(sobel_y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

  # create a normal map by stacking the gradients and a constant z value
  normal_map = np.dstack((sobel_x, sobel_y, np.full_like(gray, 127)))

  # encode the normal map to a base64 string
  _, normal_map_data = cv2.imencode('.png', normal_map)
  normal_map_base64 = base64.b64encode(normal_map_data).decode('utf-8')

  # return the normal map base64 string
  return normal_map_base64




img = input("Please enter the path to the image you want to convert to a normal map: ")

# Read the image from the specified path
image = cv2.imread(img)

# Convert to base64
image_bytes = cv2.imencode('.png', image)[1].tobytes()
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Compute the normal map
normal_map_base64 = normal_map_from_base64(image_base64)

# Save the normal map to a file
normal_map_bytes = base64.b64decode(normal_map_base64)
normal_map_array = np.frombuffer(normal_map_bytes, dtype=np.uint8)
normal_map = cv2.imdecode(normal_map_array, cv2.IMREAD_COLOR)
cv2.imwrite('normal_map.png', normal_map)

# Display the normal map in a window
cv2.imshow('Normal Map', normal_map)