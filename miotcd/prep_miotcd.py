import os
import shutil
import argparse
import pandas as pd
import random
import cv2
import yaml

def get_yolo_bbox(width, height, x1, y1, x2, y2):
    cx = ((x1+x2)/2)/width
    cy = ((y1+y2)/2)/height
    w = abs(x2-x1)/width
    h = abs(y2-y1)/height
    return cx, cy, w, h

def create_img_folder(in_dir, out_dir, imgs, df):
    # Create the necessary dirs
    if os.path.exists(out_dir): 
        shutil.rmtree(out_dir)
    img_out_dir = os.path.join(out_dir, 'images')
    label_out_dir = os.path.join(out_dir, 'labels')
    os.makedirs(img_out_dir)
    os.makedirs(label_out_dir)
    
    """
    We need to generate a `.txt` label file and the image itself, adding them to 'images' and 'labels' respectively.
    The label text content is fine. That's a simple matter of writing the contents using some simplified writer
    The bounding box calculation... is a little weird.
    - The provided data are in topleft-x, topleft-y, bottomright-x, bottomright-y, all in pixels
    - Yolo expects the data to be percentages, with x1 and y1 being the CENTER and x2 and y2 being the widtha and height respectively.
    """

    # Iterate through our images
    for img in imgs:
        # Get image details
        img_name = f'{img}.jpg'
        img_src = os.path.join(in_dir, img_name)
        img_dest = os.path.join(img_out_dir, img_name)
        frame = cv2.imread(img_src)
        height, width = frame.shape[:2] 

        # Move the image to the new location
        os.rename(img_src, img_dest)
        
        # filter our rows to the current image
        img_rows = df[df['img'] == img]
        
        # For each image...
        for index, row in img_rows.iterrows():            
            # Derive the box coordinates compatible with yolo
            cx, cy, w, h = get_yolo_bbox(width, height, row['x1'], row['y1'], row['x2'], row['y2'])
            bbox = f"{cx} {cy} {w} {h}"
            # Append the bbox data into our text file
            label_dest = img_dest.replace("images", "labels").replace("jpg", "txt")
            with open(label_dest, "a") as file:
                file.write(f"{row['label_id']} {bbox}\n")

    print(f'MOVED IMAGES TO: "{img_out_dir}"')

def convert_miotcd_to_yolo(dir, train_csv, split):
    # Define our dataframe
    df = pd.read_csv(os.path.join(dir,train_csv), 
                    names=['img','label','x1','y1','x2','y2'], 
                    dtype={
                        'img': 'string',
                        'label': 'string',
                        'x1': 'int64',
                        'y1': 'int64',
                        'x2': 'int64',
                        'y2': 'int64'
                    })
    df['row_id'] = df.index
    print(df.head())

    # Get some info about our dataframe
    classes = list(df['label'].unique())
    imgs = list(df['img'].unique())
    df['label_id'] = df['label'].apply(lambda x: classes.index(x))

    # Shuffle the dataset into training, validation, and testing data
    nimgs = len(imgs)
    random.shuffle(imgs)
    split_per = [int(nimgs * p) for p in split]
    train_imgs = imgs[:split_per[0]]
    val_imgs = imgs[split_per[0]:split_per[0]+split_per[1]]
    test_imgs = imgs[split_per[0]+split_per[1]:]

    # Generate folders for training, validation, and testing
    output_dir = os.path.join(dir, 'miotcd-yolo')
    output_train = os.path.join(output_dir, 'train')
    output_val = os.path.join(output_dir, 'validate')
    output_test = os.path.join(output_dir, 'test')
    create_img_folder(os.path.join(dir, 'train'), output_train, train_imgs, df[df['img'].isin(train_imgs)])
    create_img_folder(os.path.join(dir, 'train'), output_val, val_imgs, df[df['img'].isin(val_imgs)])
    create_img_folder(os.path.join(dir, 'train'), output_test, test_imgs, df[df['img'].isin(test_imgs)])

    # Print to yaml
    out_classes = '[' + ','.join([f"'{p}'" for p in classes]) + ']'
    out_yaml = f"""
    path: {dir}  # dataset root dir
    train: train/  # train images (relative to 'path')
    val: validate/ # val images (relative to 'path')
    test:  test/ # test images (optional)

    # Classes
    nc: {len(classes)}  # number of classes
    names: {out_classes}  # class names
    """

    out_yaml_safe = yaml.safe_load(out_yaml)
    with open(os.path.join(output_dir, 'miotcd-yolo.yaml'), 'w') as file:
        yaml.dump(out_yaml_safe, file)


    """
    val_file = os.path.join(dir,'annotations_val.csv')
    annotation_file = open(val_file)
    annotation_file = csv.reader(annotation_file, delimiter=" ")
    image_paths = os.listdir("PATH_TO\\images")

    prev_img_path = ""
    counter = 0
    for annotation in annotation_file:
        # Simple counter for progress
        if counter % 12084 == 0:
            print(counter)
        counter += 1

        # 
        annotation = annotation[0].split(",")
        img_path = "PATH_TO\\images\\" + annotation[0]
        resize_factor = 640 / max(int(annotation[6]), int(annotation[7]))

        if img_path != prev_img_path:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor)))
            cv2.imwrite(img_path, img)
            prev_img_path = img_path
        
        bbox = f"{(int(annotation[1]) * resize_factor) / (int(annotation[6]) * resize_factor)} {(int(annotation[2]) * resize_factor) / (int(annotation[7]) * resize_factor)} {(int(annotation[3]) * resize_factor) / (int(annotation[6]) * resize_factor)} {(int(annotation[4]) * resize_factor) / (int(annotation[7]) * resize_factor)}"
        x1 = (int(annotation[1]) * resize_factor)
        y1 = (int(annotation[2]) * resize_factor)
        x2 = (int(annotation[3]) * resize_factor)
        y2 = (int(annotation[4]) * resize_factor)
        width = (int(annotation[6]) * resize_factor)
        height = (int(annotation[7]) * resize_factor)
        bbox = f"{((x1 + x2) / 2) / width} {((y1 + y2) / 2) / height} {(x2 - x1) / width} {(y2 - y1) / height}"
        label_path = img_path.replace("images", "labelsval").replace("jpg", "txt")
        with open(label_path, "a") as file:
            file.write("1 " + bbox + "\n")

def check_sku_imgs():
    image_paths = os.listdir("PATH_TO\\images")
    for img in image_paths:
        if img.find("train") != -1:
            image = cv2.imread("PATH_TO\\images\\" + img)
            if image.shape[1] > 640:
                print(img)
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modify the MIO-TCD Localization dataset into a format compatible with YOLO")
    parser.add_argument('-d', '--dataset_dir', 
                        type=str, 
                        help="Relative root directory path to the Mio-TCD Localization directory",
                        default='MIO-TCD-Localization')
    parser.add_argument('--train', 
                        type=str,
                        help="The CSV file (relative to the root diretory) that contains the training image data",
                        default='gt_train.csv')
    parser.add_argument('-s', '--split',
                        nargs=2,
                        type=float,
                        help="How should the images be split?",
                        default=[0.6, 0.2])
    parser.add_argument('-r', '--reset',
                        help="Reset the images. This means moving all images in the 'miotcd-yolo' folder back into its original 'train' folder.",
                        action='store_false')
    args = parser.parse_args()

    convert_miotcd_to_yolo(args.dataset_dir, args.train, args.split)