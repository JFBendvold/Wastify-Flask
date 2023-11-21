from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import base64
from datetime import datetime

app = Flask(__name__)

# MobileNet imports
class_names = ['cardboard','food', 'glass','hazardous', 'metal', 'paper', 'plastic', 'trash']
mobilenet = torch.hub.load('pytorch/vision:v0.16.0', 'mobilenet_v3_large', weights='IMAGENET1K_V1')

# YOLO model
model = YOLO('models/yolo.pt')

# MobileNet transforms
num_classes = len(class_names) 
last_channel = mobilenet.classifier[-1].in_features
mobilenet.classifier[-1] = torch.nn.Linear(last_channel, num_classes)
mobilenet.load_state_dict(torch.load('models/mnet.pth', map_location=torch.device('cpu')))
test_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

@app.route('/process_image', methods=['POST'])
def process_image():
    predictions = []
    paths = []
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'})
    
    print("Image received")

    image_file = request.files['file']
    image = Image.open(image_file)

    # YOLO processing
    results = model(image)
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        results_image = im
    
    buffer= io.BytesIO()
    results_image.save(buffer, format="JPEG")
    image_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    predictions.append({'class': "yolo_results", 'probability': 0, 'image': image_str})

    for i, result in enumerate(results):
        boxes = result.boxes.xyxy
        for j, box in enumerate(boxes):
            dt = datetime.now()
            ts = datetime.timestamp(dt)
            x_min, y_min, x_max, y_max = map(int, box)
            region = image.crop((x_min, y_min, x_max, y_max))
            path = f'results/image_marked_resized_{i}_{j}_{ts}.jpg'
            paths.append(path)
            region.save(path)

    # MobileNet processing
    
    for path in paths:
        region_image = Image.open(path)
        image_to_show = region_image.copy()  # Create a copy to avoid modifying the original
        region_image = test_transform(region_image)
        mobilenet.eval()
        with torch.no_grad():
            region_image = region_image.unsqueeze(0)
            output = mobilenet(region_image)
            _, predicted = torch.max(output.data, 1)
            class_name = class_names[predicted.item()]
            probability = torch.max(torch.nn.functional.softmax(output[0], dim=0)).item() * 100

            # Convert image to base64-encoded string
            buffered = io.BytesIO()
            image_to_show.save(buffered, format="JPEG")
            image_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            predictions.append({'class': class_name, 'probability': probability, 'image': image_str})

    return jsonify(predictions)

@app.route('/YOLO', methods=['POST'])
def YOLO():
    if 'file' not in request.files:
        print("No image provided")
        return jsonify({'error': 'No image provided'})

    print("Image received")
    image_file = request.files['file']
    image = Image.open(image_file)

    # YOLO processing
    results = model(image)

    print("YOLO processing started")

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        #im.show()  # show image
        im.save('results1.jpg')  # save image
        result_image = im

    print("YOLO processing complete")
    #image.show()

    # Convert PIL image to byte array
    imgByteArr = io.BytesIO()
    result_image.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()

    # Return byte array
    return send_file(io.BytesIO(imgByteArr), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)