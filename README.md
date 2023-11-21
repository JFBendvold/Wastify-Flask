Wastify-Flask
Overview
Wastify-Flask is a Flask-based web application designed to process images using deep learning models for object detection and classification. It leverages two primary models: YOLO (You Only Look Once) for object detection and MobileNet for object classification.

Features
Image Processing Endpoint: Accepts images through a /process_image endpoint, performing object detection and classification.
YOLO Object Detection: Utilizes the YOLO model to detect objects in the image.
MobileNet Classification: Classifies detected objects into predefined categories using the MobileNet model.
Image Saving and Conversion: Saves detected objects as images and converts them to base64-encoded strings for easy transmission.
Setup
Clone the Repository: git clone https://github.com/JFBendvold/Wastify-Flask
Install Dependencies: Install necessary Python libraries including Flask, PIL, torch, torchvision, and others as needed.
Model Setup: Ensure the YOLO model file (yolo.pt) and MobileNet state dictionary (mnet.pth) are placed in the models directory.
Run the Application: Start the Flask server with python main.py.
API Endpoints
POST /process_image: Accepts an image file and returns the processed results including object classifications and probabilities.
POST /YOLO: Accepts an image file and returns the YOLO processed image.
Requirements
Python 3.x
Flask
PyTorch
torchvision
PIL (Python Imaging Library)
ultralytics YOLO library
numpy
Note
This application is designed for educational and demonstration purposes. Accuracy and performance depend on the models and training data used.
