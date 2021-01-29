#coding: utf-8
import glob
import os
import numpy as np
import pickle
import multiprocessing
import ctypes
import sys

if len(sys.argv)>=3:
    base_path=sys.argv[1]
    base_path2=sys.argv[2]
else:
    base_path = '/media/disk3/miccai-colon/neg-output1/'    #input patch with common images
    base_path2 = '/media/disk2/miccai-colon/neg-output1/'               #input path with predict

input_path=base_path+'common_images/'
input_path2=base_path2+'predict/'
output_path=base_path2+'heatmap_V3/'

if os.path.exists(output_path):
    os.system('rm -rf '+output_path)

os.mkdir(output_path)

input_cases=glob.glob(input_path+'*')
heatmap_data={}

fw=open(output_path+'case_level_feature','w')    #write feature from predict
workers = 3

def compute_max_prob(heatmap, step, height, width, max_prob):
    for ii in range(0, height - step):
        for jj in range(0, width - step):
            mean_prob = np.mean(heatmap[ii:ii + step, jj:jj + step])

            if mean_prob > max_prob[step - 2]:
                max_prob[step - 2] = mean_prob

share_data = multiprocessing.Array(ctypes.c_float, 3)
max_prob = np.frombuffer(share_data.get_obj(), dtype=np.float32)

print('begin read prediction data')
for case in input_cases:
    casebasename=os.path.basename(case)

    if casebasename=='1116037':
        a=0

    print('begin process: '+casebasename)

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

        all_count+=1

    if all_count==0:
        continue

    heatmap[heatmap<0.5]=0
    heatmap_data[casebasename]=heatmap
    height, width=heatmap.shape

    #get 10X, 7X, 5X image mean and max probability (2*2,3*3,4*4 block)
    max_prob[0]=0   #must init by one-by-one, or max_prob is a new element
    max_prob[1]=0
    max_prob[2]=0
    mag_size = [2, 3, 4]
    process = []
    for i in range(workers):
        p = multiprocessing.Process(target=compute_max_prob,args=(heatmap, mag_size[i],height,width, max_prob))
        process.append(p)

    for i in range(workers):
        process[i].start()

    for i in range(workers):
        process[i].join()

    fw.writelines(casebasename + '(10X, 7X and 5X) case level positive probability:  %.3f, %.3f, %.3f\n' % (max_prob[0], max_prob[1], max_prob[2]))

fw.close()
print('begin save heatmap')
fw = open(output_path+'heatmap', 'w')
pickle.dump(heatmap_data, fw)
fw.close()
print('finish save heatmap')



