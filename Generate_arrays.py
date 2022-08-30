#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
import cv2

hr_images=[]
lr_images=[]

k=0
n=3000

for img in os.listdir("/home/wytcor/Desktop/STAGE/YOLOv5_Custom_Training/datasets/DOTAV15/trainsplit/images"):
    img_array = cv2.imread("/home/wytcor/Desktop/STAGE/YOLOv5_Custom_Training/datasets/DOTAV15/trainsplit/images/"+ img)

    hr_img_array = cv2.resize(img_array, (416,416))
    lr_img_array = cv2.resize(img_array,(104,104))
    lr_images.append(lr_img_array/255.)
    hr_images.append(hr_img_array/255.)
    k=k+1
    del hr_img_array
    del lr_img_array
    print(k)
    if k==n:
        break
    else:
        pass


# In[ ]:


from numpy import save,load

save('data_lr_image_array.npy', lr_images)
save('data_hr_image_array.npy', hr_images)

del hr_images
del lr_images


# In[ ]:


data_lr_image_array = load('data_lr_image_array.npy')
data_hr_image_array = load('data_hr_image_array.npy')

