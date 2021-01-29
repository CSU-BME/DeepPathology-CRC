#coding:utf-8
import pickle
import os
import cv2
import numpy as np
import sys

if len(sys.argv)>=3:
    base_path = sys.argv[1]
    base_path2 = sys.argv[2]
else:
    base_path = '/media/disk6/pathology_data_process_5X/group1/'
    base_path2 = base_path

blocksize = 30     #for 20X magnification
if len(sys.argv)==4:
    blocksize = 120   #only for 5X

heatmap_pathname=base_path2+'heatmap/heatmap'
thumbnail=base_path+'preview_images/'
output_images=base_path2+'heatmap_images/'

if os.path.exists(output_images):
    os.system('rm -rf '+output_images)

os.mkdir(output_images)
fw=open(heatmap_pathname,'r')
heatmap_data=pickle.load(fw)

for basename,heatmap in heatmap_data.iteritems():
    print('begin process '+basename)
    pathname=thumbnail+basename+'.jpg'

    if not os.path.exists(pathname):
        print('the image file:'+basename+'doesn\'t exist')
        continue

    if basename=='1130788  1,2':
        a=0

    img=cv2.imread(pathname)
    height,width=img.shape[0],img.shape[1]
    del img

    img=np.zeros((height,width,3),dtype=np.uint8)
    heat_height,heat_width=heatmap.shape[0],heatmap.shape[1]

    stepy=0
    stepx=0
    for i in range(0,height,blocksize):
        for j in range(0,width,blocksize):
            img[i:min(i+blocksize,height),j:min(j+blocksize,width),:]=np.uint8(heatmap[min(stepy,heat_height-1),min(stepx,heat_width-1)]*255)
            stepx+=1
        stepx=0
        stepy+=1

    cv2.imwrite(output_images+basename+'_heatmap.jpg',img)


fw.close()


