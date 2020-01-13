import os
import random
import cv2
import numpy as np
from tqdm import tqdm

def roll_image():

    path = '/home/xuan/data/QMUL-SurvFace/training_set/'

    roll = random.sample(os.listdir(path), 2)
    img1 = random.choice(os.listdir(path + roll[0]))
    img2 = random.choice(os.listdir(path + roll[1]))
    
    return img1, path + roll[0] + '/' + img1, img2, path + roll[1] + '/' + img2

def concatenate(img1, path_1, img2, path_2):

    save_path = '/home/xuan/exdisk/LR-Experiment-Data/data_roll_20W/0/'    

    img1_ = cv2.imread(path_1)
    img2_ = cv2.imread(path_2)
  
    if img1_ is None or img2_ is None:
        print(path_1 + '\n' ,path_2)
        return 1
        
    img1_ = cv2.resize(img1_, (20, 20), interpolation=cv2.INTER_CUBIC)
    img2_ = cv2.resize(img2_, (20, 20), interpolation=cv2.INTER_CUBIC)
        
    img_merge = np.concatenate((img1_, img2_), axis = 1)
   
    cv2.imwrite(save_path + img1 + '+' + img2, img_merge)

def main():
    count = 40
    pbar = tqdm(total = count)
    while(1):
        img1, img1_path, img2, img2_path = roll_image()
        r =  concatenate(img1, img1_path, img2, img2_path)
        count -= 1
        pbar.update(1)
        if count == 0:
            pbar.close()
            break

if __name__ == '__main__':
    main()
