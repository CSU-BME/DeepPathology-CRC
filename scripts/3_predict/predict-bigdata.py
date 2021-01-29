#coding:utf-8
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import os
import pickle
import time
import sys
import cv2
import numpy as np

sys.path.append("../slim")
slim = tf.contrib.slim

from nets import inception
from preprocessing import inception_preprocessing

# os.environ['CUDA_VISIBLE_DEVICES'] = '' #Uncomment this line to run prediction on CPU.

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)
ex_flag=False    #exchage labels when output results

def get_test_images(mypath):
     return [mypath + '/' + f for f in listdir(mypath) if isfile(join(mypath, f)) and (f.find('.jpg') != -1  or f.find('.jpeg') != -1 )]

def transform_img_fn(image_file,patch_mask,beg_height,beg_width,heinum,widnum, train_mean=(0.69,0.53,0.72)):
    images = []
    patch_name=[]

    image_raw=cv2.imread(image_file)
    image_raw = image_raw[:, :, (2, 1, 0)]


    for ii in range(heinum):
        for jj in range(widnum):
            if patch_mask[beg_height+ii,beg_width+jj]==False:
                continue

            #get center region
            patch_raw = image_raw[ii * sampleheight+offsethei:(ii + 1) * sampleheight-offsethei, jj * samplewidth+offsetwid:(jj + 1) * samplewidth-offsetwid, :]
            patch_raw = ((patch_raw / 255.0)-0.5)*2

            #save image data
            images.append(patch_raw)
            patch_name.append([str(beg_height + ii), str(beg_width + jj)])

    return images,patch_name


if __name__ == '__main__':

    if len(sys.argv) < 6:
        print("The script needs five arguments.")
        print("The first argument should be the CNN architecture: v1, v3 or inception_resnet2")
        print("The second argument should be the directory of trained model.")
        print("The third argument should be directory of test images.")
        print("The  fourth argument should be output file for predictions.")
        print("The  fifth argument should be number of classes.")
        exit()
    deep_lerning_architecture = sys.argv[1]
    train_dir = sys.argv[2]
    test_path = sys.argv[3]
    output = sys.argv[4]
    nb_classes = int(sys.argv[5])

    if len(sys.argv)==7:
        ex_flag=True if sys.argv[6].lower()=='true' else False
        print('ex_flag is True, exchange output labels')

    # get image info
    basename = os.path.basename(test_path)
    print('Process the case: ' + basename + ' at  ' + time.asctime(time.localtime(time.time())))
    fto = open(output, 'w')

    test_batch = 500
    print('the batch patches size  is ' + str(test_batch))

    fw_info = open(os.path.join(test_path, basename + '_info'), 'r')
    split_height = pickle.load(fw_info)  # image count at height
    split_width = pickle.load(fw_info)  # image count at width
    sampleheight = pickle.load(fw_info)  # patch height
    samplewidth = pickle.load(fw_info)  # patch width
    patch_mask = pickle.load(fw_info)  # save mask of patches and background
    saved_dict = pickle.load(fw_info)

    central_fraction = 0.875    #crop center region from patches  >0 and <1
    offsetwid=int(samplewidth * (1 -central_fraction)/2)
    offsethei=int(sampleheight*(1-central_fraction)/2)

    inimage_height=sampleheight-2*offsethei
    inimage_width=samplewidth-2*offsetwid

    if deep_lerning_architecture == "v1" or deep_lerning_architecture == "V1":
        image_size = 224
    else:
        if deep_lerning_architecture == "v3" or deep_lerning_architecture == "V3" or deep_lerning_architecture == "v4" or deep_lerning_architecture == "V4" or deep_lerning_architecture == "resv2" or deep_lerning_architecture == "inception_resnet2":
            image_size = 299
        else:
            print("The selected architecture is not correct.")
            exit()

    image_list = get_test_images(test_path)
    new_images = tf.placeholder(tf.float32, shape=(None, inimage_height, inimage_width, 3))
    processed_images = tf.image.resize_bilinear(new_images, [image_size, image_size], align_corners=False)

    if deep_lerning_architecture == "v1" or deep_lerning_architecture == "V1":
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(processed_images, num_classes=nb_classes, is_training=False)

    else:
        if deep_lerning_architecture == "v3" or deep_lerning_architecture == "V3":
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                logits, _ = inception.inception_v3(processed_images, num_classes=nb_classes, is_training=False)
        else:
            if deep_lerning_architecture == "resv2" or deep_lerning_architecture == "inception_resnet2":
                with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
                    logits, _ = inception.inception_resnet_v2(processed_images, num_classes=nb_classes,
                                                              is_training=False)

    if deep_lerning_architecture == "v4" or deep_lerning_architecture == "V4":
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v4(processed_images, num_classes=nb_classes, is_training=False)

    def predict_fn(images):
        return session.run(probabilities, feed_dict={new_images: images})

    probabilities = tf.nn.softmax(logits)
    checkpoint_path = tf.train.latest_checkpoint(train_dir)
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore())
    init_fn(session)

    print('the images files in the case are: '+str(split_width*split_height))

    count=0
    beg_width=0    #beginning patch index at width
    beg_height=0
    heinum=0
    widnum=0
    tissue_patch=[]
    patch_list=[]
    for ii in range(split_height):    #process all the image files
        for jj in range(split_width):
            print('begin read and transform the  %d image file'%(count+1))
            saved_name = basename + '_' + str(ii) + '_' + str(jj) + '.jpg'

            [heinum,widnum]=saved_dict[saved_name].split('_')
            heinum=int(heinum)/sampleheight   #patches num at height
            widnum=int(widnum)/samplewidth

            #read image data and tissue patches list
            temp_patch,temp_list=transform_img_fn(os.path.join(test_path,saved_name),patch_mask,beg_height,beg_width,heinum,widnum)

            tissue_patch.extend(temp_patch)
            patch_list.extend(temp_list)
            count += 1  # image count
            beg_width = beg_width + widnum

        beg_height = beg_height + heinum
        beg_width=0

    test_num = len(patch_list) / test_batch + 1

    import time
    time_start = time.time()

    for batch_i in range(test_num):
        #process patches in one image file, one batch one times
        print('Start doing predictions: the ' + str(batch_i + 1) + '  batch')
        preds = predict_fn(tissue_patch[batch_i*test_batch:(batch_i+1)*test_batch])   #get possibility of patches in the batch

        for p in range(len(preds)):
            image_info=basename+'_'+patch_list[p + batch_i * test_batch][0]+'_'+patch_list[p + batch_i * test_batch][1]
            fto.write(image_info)    #write patch indexes
            if ex_flag==False:
               for j in range(len(preds[p, :])):
                    fto.write('\t' + str(preds[p, j]))
            else:
                for j in reversed(range(len(preds[p, :]))):
                    fto.write('\t' + str(preds[p, j]))

            fto.write('\n')

    time_end = time.time()
    print('the time cost is ************************************************:', time_end-time_start)


    fw_time=open('timesaved','a')
    fw_time.writelines(str(time_end-time_start)+'\n')
    fw_time.close()

    fto.close()
    fw_info.close()
    print('Finish the case: ' + basename + ' at  ' + time.asctime(time.localtime(time.time())))
