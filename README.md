# Meta Quest Pro Object Tracking

_This repository handles object detection on [footage captured from the Meta Quest Pro](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-SCRCPY.git) and [corrected for lens distortion](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-LensCorrection.git). We will use YOLO v8 + the MIO-TCD Localization Dataset for object classification and detection._

## Software Requirements

### Python Environment

It is HIGHLY recommended to run this in a virtual environment. If you do not know how to do this, follow these instructions.

Note that if you want to do any training and prediction using the GPU on Windows, you will need Python `3.8` - `3.11`; anything outside this range is not compatible with CUDA and PyTorch.

1. If needed, install any python version between `3.8` and `3.11`. This can be done using python's installer packaging and doesn't need any fancy code work. However, to confirm if you have the python version installed and accessible in your system, you can check via:

````bash
py --list
py -0
````

2. Install **virtualenv** using your preferred Python version. For example, to install with Python version `3.11`, you can type:

```bash
py -3.11 -m pip install virtualenv
```

3. Initialize the new python virtual environment:

````bash
py -3.11 -m virtualenv <ENVIRONMENT NAME>
````

4. Activate the new environment:

````bash
<ENVIRONMENT NAME>\Scripts\activate
````

5. Call the following command to install all necessary packages. This wil install `ultralytics`, which contains commands for YOLO and other needs.

````bash
pip install -r requirements.txt
````

6. Separately from `requirements.txt`, install the neceesary CUDA-compatible `torch` packages via command line. This command will change depending on your operating system and CUDA version. To check, you can follow the instructions [on this webpage](https://pytorch.org/).

````bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
````

### Git Cloning + Ultralytics

This repository contains a single submodule. To ensure that this submodule is installed alongside this repo, please perform the following `git clone` command:

````bash
git clone https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-ObjectTracking.git --recursive
````

If you omitted `--recursive` from the clone command, you can actually still get the submodule repos via:

````bash
git submodule update --init --recursive
````

However, there is the possibility that attempting to import `ultralytics` on any Python version lower than `3.12` may be unsuccessful. To ensure that `ultralytics` can be imported properly, follow these instructions:

1. Clone the ultralytics repository
````bash
git clone https://github.com/ultralytics/ultralytics
````

2. Navigate to the cloned directory
````bash
cd ultralytics
````

3. Install the package in editable mode for development (don't forget the `.` at the end!)
````bash
pip install -e .
````

## Using YOLO + MIO+TCD

It is highly recommended to [follow along this document from Ultralytics](https://docs.ultralytics.com/usage/python/) on how to use YOLOv8. It's not that difficult, honestly speaking.

Where it gets interesting is training a pre-trained YOLOv8 model with the MIO-TCD Localization dataset, which can be found [here](https://tcd.miovision.com/). We need to be able to train a YoloV8 model, then perform transfer learning on that trained model with MIO-TCD.

### Training a YOLOv8 Model

Honestly, just take the code from the official Ultralytics webpage on python usage.

### Preparing MIO-TCD

We are using the **MIO-TCD Localization Dataset** to train a **YOLO v8** model to detect urban elements. However, this step is much more complicated because the MIO-TCD Localization dataset needs to be modified in file structure!

The MIO-TCD Localization dataset contains the following (relevant) contents:

```
├── train/
│   ├── 000000.jpg
│   ├── ...
├── test/
│   ├── 080000.jpg
│   ├── ...
├── gt_train.csv
├── your_result_test.csv
├── your_result_train.csv
```

 According to MIO-TCD, there are a total of 110000 training images and some other number of test images. However, the test images do not come with any labels, so we can't use them necessarily.

 YOLO expects the folder structure of any datasets to contain the following folder arrangement:

 ```
├── train/
│   ├── images/
│   │   ├── 000000.jpg
│   │   ├── ...
│   ├── labels/
│   │   ├── 000000.txt
│   │   ├── ...
├── validate/
│   ├── images/
│   │   ├── 000000.jpg
│   │   ├── ...
│   ├── labels/
│   │   ├── 000000.txt
│   │   ├── ...
├── test/
│   ├── images/
│   │   ├── 000000.jpg
│   │   ├── ...
│   ├── labels/
│   │   ├── 000000.txt
│   │   ├── ...
...
```

Okay, they don't have to be named exactly `train/`, `test/`, or `validate/`. In fact, we need a new `.yaml` file that contains the following details:

* `path`: The root directory. Can be relative or absolute... though I don't know where the relativity starts from...
* `train`: The directory, relative to `path`, that contains the training images
* `val`: The directory, relative to `path`, that contains the validation images
* `test (OPTIONAL)`: The directory, relative to `path`, that contains the test images
* `nc`: The number of classes
* `names`: The individual classes found among all dataset images

So not only do we have to rearrange the files but also we have to generate a new `.yaml` file that will tell YOLO where to find the necessary datset images for training, validation, and possibly testing. 

1. Call the following python script:

````bash
python miotcd/prep_miotcd.py -d <DATASET_DIR> --train <PATH/TO/>gt_train.csv -s <VAL B/W 0 and 1, TRAINING DATASET %> <VAL B/W 0 and 1, VALIDATION DASASET %> 
````

2. Go into the newly generated `miotcd-yolo.yaml` file and modify `path` so that it is an ABSOLUTE directory, rather than an absolute directory. **YOU HAVE TO DO THIS EVERY TIME YOU RELOCATE THE DATASET**.

### Train your YOLOv8 Model on Coco

### Training a Pre-Trained YOLOv8 Model

After re-arranging the MIO-TCD dataset, you can begin training. Simply call this command:

````bash
python miotcd/train.py -m <PATH/TO/PRETRAINED/YOLOv8/.pt FILE> -d <PATH/TO/>miotcd-yolo.yaml -e <NUMBER OF EPOCHS> --mac 
````

The last function call, `--mac`, can only be called if you are on a Silicon Macbook (either M1 or M2) and have OS X version `13.0.0` or greater. If you're on Windows, omit this flag.

The model will attempt to train using the GPU. You can modify this behavior by changing `miotcd/train.py` directly to suit your needs. Note that for Windows, you will need to use `workers=0` for any GPU-related tasks. On Mac, you need `devices=____` instead - feel free to look it up in the Ultralytics documentation.

### Continuing the Training of your MIO-TCD + YOLOv8 Model

If you find that you need to continue training the model further beyond your original number of epochs, you can do the following:

* Make sure you have the results of the last training session saved. This includes having access to the `last.pt` weights file.
* Rather than refer to `yolov8.pt` (or whatever you used in the initial training command), refer ot this `last.pt` file instead
* In the python code `miotcd/train.py`, make sure to add `petrained=True` to the command that initializes training. This prevents any confusion from the model regarding the completed number of epochs (if not included, an error will be printed stating something like all epochs have already been trained and the training will not commence).

### Using the MIO-TCD + Yolov8 Model

To perform the object detection, make sure to use the following:

````bash
python miotcd/predict.py <PATH/TO/INPUT/VIDEO> -m <PATH/TO/.pt FILE> -g
````

The last flag, `-g`, corresponds with Windows and CUDA-enabled GPU usage, like with `train.py`. Again, if you have a Macbook, you can replace this and the code with the necessary changes.

This last step will add bounding boxes to the footage and generate a CSV file that contains the bounding boxes for each object in each timestamp of the image.