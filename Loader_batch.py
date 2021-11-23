import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        if img is not None:
            images.append(img)
            
    return images

def data_crop(img,crop_size):
    ims = []
    for im in img:
        h, w = im.shape
        rand_range_h = h-crop_size
        rand_range_w = w-crop_size
        x_offset = np.random.randint(rand_range_w)
        y_offset = np.random.randint(rand_range_h)
        im = im[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
        ims.append(im)
        
    return ims

def data_augment(img):
    ims = []
    for im in img:
        h, w = im.shape
        if np.random.rand() > 0.5:
            im = np.fliplr(im)
        if np.random.rand() > 0.5:
            angle = 10*np.random.rand()
            if np.random.rand() > 0.5:
                angle *= -1
            M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
            im = cv2.warpAffine(im,M,(w,h))
        ims.append(im)
        
    return ims
