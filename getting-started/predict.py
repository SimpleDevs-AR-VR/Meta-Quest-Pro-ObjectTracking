from ultralytics import YOLO
from PIL import Image

model = YOLO('runs/detect/train/weights/best.pt')

test_img = Image.open('./getting_started/bus.jpg')
results = model.predict(source=test_img, save=True)