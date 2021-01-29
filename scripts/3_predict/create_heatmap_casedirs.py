#coding: utf-8
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import sys

if len(sys.argv)>=3:
    base_path=sys.argv[1]
    base_path2=sys.argv[2]
else:
    base_path = '/media/disk6/pathology_data_process_5X/group1/'
    base_path2=base_path

input_path=base_path+'common_images/'
input_path2=base_path2+'predict/'
output_path=base_path2+'heatmap/'

distance=2

if os.path.exists(output_path):
    os.system('rm -rf '+output_path)

os.mkdir(output_path)

input_cases=glob.glob(input_path+'*')
heatmap_data={}

fw=open(output_path+'pos_percent','w')
fw2=open(output_path+'cluster_value','w')


print('begin read prediction data')
for case in input_cases:
    casebasename=os.path.basename(case)

    fw_info = open(os.path.join(case, casebasename + '_info'), 'r')
    split_height = pickle.load(fw_info)  # image count at height
    split_width = pickle.load(fw_info)  # image count at width
    sampleheight = pickle.load(fw_info)  # patch height
    samplewidth = pickle.load(fw_info)  # patch width
    patch_mask = pickle.load(fw_info)  # save mask of patches and background
    saved_dict = pickle.load(fw_info)

    heatmap=np.zeros((patch_mask.shape[0],patch_mask.shape[1]))

    #get pos prob
    pathname=input_path2+casebasename+'.txt'

    if casebasename=='1130788  1,2':
        a=0

    predicted=open(pathname).readlines()

    all_count=0
    pos_count=0

    for line in predicted:
        predictinfo=line.strip().split()

        pos_value=float(predictinfo[-1])

        temp=predictinfo[-3]     #get posistion info
        #temp=temp[:-4]         #remove filename extension

        position_info=temp.strip().split('_')

        heatmap[int(position_info[-2]),int(position_info[-1])]=float(pos_value)

        if float(pos_value)>0.5:
            pos_count+=1

        all_count+=1

    if all_count==0:
        continue

    heatmap[heatmap<0.5]=0
    #plt.figure()
    #plt.imshow(heatmap,cmap=plt.cm.hot)
    #plt.show()

    heatmap_data[casebasename]=heatmap
    fw.writelines(casebasename+'  pos_count, all_count and percent: %d, %d, %.2f \n'%(pos_count, all_count, float(pos_count)/all_count))

    filter_heatmap=np.uint8(heatmap* 255)
   # filter_heatmap=cv2.medianBlur(filter_heatmap, distance)
    ret, filter_heatmap=cv2.threshold(filter_heatmap, 127, 1, cv2.THRESH_BINARY)
    kernel = np.ones((distance, distance), np.uint8)
    filter_heatmap = cv2.erode(filter_heatmap, kernel, iterations=1)

    cluster_count=filter_heatmap[filter_heatmap>0.5].shape[0]
    fw2.writelines(casebasename + '  pos_count, all_count and cluster_count: %d, %d, %d \n' % (pos_count, all_count, cluster_count))

fw.close()
print('begin save heatmap')
fw = open(output_path+'heatmap', 'w')
pickle.dump(heatmap_data, fw)
fw.close()
print('finish save heatmap')

fw2.close()









