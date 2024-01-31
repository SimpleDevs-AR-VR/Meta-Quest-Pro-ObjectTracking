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
    args = parser.parse_args()

    convert_miotcd_to_yolo(args.dataset_dir, args.train, args.split)