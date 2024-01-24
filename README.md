# Meta Quest Pro Object Tracking

_This repository handles object detection on [footage captured from the Meta Quest Pro](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-SCRCPY.git) and [corrected for lens distortion](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-LensCorrection.git). We will use YOLO for object classification and detection._

## Software Requirements

It is HIGHLY recommended to run this in a virtual environment. If you do not know how to do this, follow these instructions:

1. Install `virtualenv` into your system.
2. Call the following commands:

````bash
python -m virtualenv <ENVIRONMENT NAME>
<ENVIRONMENT NAME>\Scripts\activate
````

3. Call the following command to instal all necessary packages:

````bash
pip install -r requirements.txt
````

