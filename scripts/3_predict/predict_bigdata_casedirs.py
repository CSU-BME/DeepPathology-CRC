#coding:utf-8
import os
import glob
import sys

base_path=sys.argv[1]
output_base_path=sys.argv[2]
model_path=sys.argv[3]

input_path=base_path+'common_images/'
output_path=output_base_path+'predict/'

dirs=glob.glob(input_path+'*')

if os.path.exists(output_path):
    print('Error: the output_path has exist, please make sure a new path')
    exit()

os.mkdir(output_path)

print('There are '+str(len(dirs))+' dirs')
for line in dirs:
    imags=glob.glob(line+'/*')
    if len(imags)==0:
        print('no image file at '+line)
        continue

    basename=os.path.basename(line)
    predictfile=output_path  + basename + '.txt'

    if os.path.exists(predictfile):
        print(predictfile+' has existed')
        continue

    basecommand = 'python predict-bigdata.py  v3  ' + model_path + ' "' + line + '"  ' + output_path + '"' + basename + '.txt"   2 True'
    print('processing '+line+' ********************')
    os.system(basecommand)
    print('\n')


