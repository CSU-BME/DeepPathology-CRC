#pathes-level prediction for whole slide images.
import os
import time

input_base_path = '/media/disk2/pathology_data_process/'       #parent dirs of groupX with common images and previews
group_dirs=['group1']    #data dirs of whold slide images

output_base_path='../../output/predicted/result_1/'
model_path='../../output/models/predefined_model_1/'

for group in group_dirs:
    input_group_path=input_base_path+group+'/'
    output_group_path=output_base_path+group+'/'

    if not os.path.exists(input_group_path):
        print('wrong input path')
        continue

    if os.path.exists(output_group_path):
        print('group output path has existed')

        if output_base_path != input_base_path:
            continue
    else:
        os.makedirs(output_group_path)

    print('Processing ' + group + ' ******************** at ' + time.asctime(time.localtime(time.time())))
    print('Building prediction *************')
    basecommand = 'python predict_bigdata_casedirs.py '+input_group_path+' '+output_group_path+' '+model_path
    os.system(basecommand)
    print('\n')

    #handle V3
    print('Begining create heatmap V3*************')
    basecommand='python create_heatmap_casedirs_V3.py '+input_group_path+' '+output_group_path
    os.system(basecommand)
    print('\n')

    print('Begining heatmap to images V3')
    basecommand='python heatmap-to-image-casedirs_V3.py '+input_group_path+' '+output_group_path
    os.system(basecommand)
    print('\n')

    #handle V2
    print('Begining create heatmap *************')
    basecommand='python create_heatmap_casedirs.py '+input_group_path+' '+output_group_path
    os.system(basecommand)
    print('\n')

    print('Begining create heatmap *************')
    basecommand = 'python create_heatmap_casedirs_3x3.py ' + input_group_path + ' ' + output_group_path
    os.system(basecommand)
    print('\n')

    print('Begining heatmap to images')
    basecommand='python heatmap-to-image-casedirs.py '+input_group_path+' '+output_group_path
  #  os.system(basecommand)
    print('\n')

