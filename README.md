# Meta Quest Pro Object Tracking

_This repository handles object detection on [footage captured from the Meta Quest Pro](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-SCRCPY.git) and [corrected for lens distortion](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-LensCorrection.git). We will use YOLO for object classification and detection._

## Software Requirements

### Git Cloning Properly

This repository contains submodules. To ensure that all submodules are installed alongside this repo, please perform the following `git clone` command:

````bash
git clone https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-ObjectTracking.git --recursive
````

If you omitted `--recursive` from the clone command, you can actually still get the submodule repos via:

````bash
git submodule update --init
````

### Python Environment

It is HIGHLY recommended to run this in a virtual environment. If you do not know how to do this, follow these instructions:

1. Install **virtualenv** into your system.
2. Call the following commands:

````bash
python -m virtualenv <ENVIRONMENT NAME>
<ENVIRONMENT NAME>\Scripts\activate
````

3. Call the following command to instal all necessary packages:

````bash
pip install -r requirements.txt
````

This wil install `ultralytics`, which contains commands for YOLO and other needs.

## Using YOLO (Basics)

It is highly recommended to [follow along this document from Ultralytics](https://docs.ultralytics.com/usage/python/) on how to use YOLOv8. It's not that difficult, honestly speaking.




