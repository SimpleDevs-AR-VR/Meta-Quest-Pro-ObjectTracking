from ultralytics import YOLO
import argparse

# PARSE ARGUMENTS
# There is 1 required argument and 1 optional argument
parser = argparse.ArgumentParser(
                    prog='Meta Quest Pro Object Detector',
                    description='This program predicts objects found in footage captured from the Meta Quest Pro.',
                    epilog='Only use after extracting footage data using scrcpy and correcting the lens distortion usinng ffmpeg')
parser.add_argument('input',
                    help="What video, image, or folder of video and/or images should we run the prediction on?")
parser.add_argument('-m', '--model', 
                    help="What model should we load?", 
                    default="yolov8n.pt")
args = parser.parse_args()

# Define the model
model = YOLO(args.model)

# Make the predictions
results = model.predict(source=args.input, save=True, save_conf=True)
