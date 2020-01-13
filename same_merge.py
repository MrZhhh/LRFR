import os
import random
import cv2
import numpy as np
from tqdm import tqdm

def roll_image():

    path = '/home/xuan/data/QMUL-SurvFace/training_set/'
    roll_dir = random.choice(os.listdir(path))
    roll = random.sample(os.listdir(path + roll_dir), 2)
    img1 = roll[0]
    img2 = roll[1]   
    return img1, path + roll_dir + '/' + roll[0], img2, path + roll_dir + '/' + roll[1]

def concatenate(save_path, img1, path_1, img2, path_2):

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
    save_path = '/home/xuan/exdisk/LR-Experiment-Data/data_roll_20W/1/'
    count = 200000 #zong shu
   
    initial_n = 100
    print("Initial Processing..." + '\n')
    pbar = tqdm(total = initial_n)
    while(initial_n):
        pbar.update(1)
        initial_n -= 1
        img1, img1_path, img2, img2_path = roll_image()
        concatenate(save_path, img1, img1_path, img2, img2_path)
    pbar.close()
    print('\n')
    print('Initial Processing Finished.')
    print('\n')       
    
    img_num = len(os.listdir(save_path))
    pbar = tqdm(total = count - len(os.listdir(save_path)))
    while(1):
        img1, img1_path, img2, img2_path = roll_image()
        concatenate(save_path, img1, img1_path, img2, img2_path)
        if len(os.listdir(save_path)) - img_num == 1:
            pbar.update(1)
            img_num += 1
        if count - len(os.listdir(save_path)) == 0:
            pbar.close()
            break

if __name__ == '__main__':
    main()
