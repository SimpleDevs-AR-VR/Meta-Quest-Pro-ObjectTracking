from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model',
                    type=str,
                    help='The model to build the YOLO instance around.',
                    default='yolov8n.pt')
parser.add_argument('-d', '--data',
                    type=str,
                    help="The dataset that ought to be used. Refers to a '.yaml' file.", 
                    default='coco128.yaml')
parser.add_argument('-e','--epochs',
                    help='How many epochs should we train under?',
                    nargs='?',
                    type=int,
                    default=3)
parser.add_argument('--mac',
                    help="Are we on an M1 or M2 mac? If so, we can optimize",
                    action='store_true')
args = parser.parse_args()

# Load a pre-trained YOLO model
model = YOLO(args.model)

# Train the model using the `coco128.yaml` dataset for 3 epochs
if args.mac:
    results = model.train(data=args.data, epochs=args.epochs, device='mps')
else:
    results = model.train(data=args.data, epochs=args.epochs, workers=0, pretrained=True)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format='onnx')