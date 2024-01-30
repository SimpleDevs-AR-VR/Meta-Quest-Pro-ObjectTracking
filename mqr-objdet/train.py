from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model',help='The model to build the YOLO instance around.',default='yolov8n.yaml')
parser.add_argument('-e','--epochs',help='How many epochs should we train under?',nargs='?',type=int,default=3)
args = parser.parse_args()

# Load a pre-trained YOLO model
model = YOLO(args.model)

# Train the model using the `coco128.yaml` dataset for 3 epochs
results = model.train(data='coco128.yaml', epochs=args.epochs)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format='onnx')