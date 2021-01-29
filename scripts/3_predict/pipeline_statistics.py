#patient-level prediction

import os
from sklearn import metrics
input_base_path = '/media/disk2/pathology_data_process/'       #parent dirs of groupX with common images and previews
group_dirs=['group1']    #data groups in pathology_data_process
output_base_path='../../output/predicted/result_1/'
output_results_tag=['xiangya']
statistics_method='EACH'   #'EACH' for one by one group, 'ALL' for all group
output_results_path=output_base_path       #output dirs of statistics results, maybe output_base_path
input_real_file='../xiangya-real.txt'    #label file

if statistics_method=='EACH':
    if len(group_dirs)!=len(output_results_tag):
        print('EACH statistics method, group number must be equal to tags')
        exit()

if statistics_method == 'ALL':
    if not isinstance(output_results_tag,str):
        print('ALL statistics method, tag is unique')
        exit()

flag=True #if Ture,  recompute results based on prediction
if flag==True:
   for group in group_dirs:
      input_group_path = input_base_path + group + '/'
      if not os.path.exists(input_group_path):
          print('wrong input path')
          continue

      output_group_path = output_base_path + group + '/'

      print('Process:  '+group +'***************')
      # handle V3
      print('Begining create heatmap V3*************')
      basecommand = 'python create_heatmap_casedirs_V3.py ' + input_group_path + ' ' + output_group_path
      #os.system(basecommand)
      print('\n')

      print('Begining heatmap to images V3')
      basecommand = 'python heatmap-to-image-casedirs_V3.py ' + input_group_path + ' ' + output_group_path  # +' True'  #True only for 5X
      #os.system(basecommand)
      print('\n')

      # handle V2
      print('Begining create heatmap *************')
      basecommand = 'python create_heatmap_casedirs.py ' + input_group_path + ' ' + output_group_path
      #os.system(basecommand)
      print('\n')

      print('Begining create heatmap *************')
      basecommand = 'python create_heatmap_casedirs_3x3.py ' + input_group_path + ' ' + output_group_path
      #os.system(basecommand)
      print('\n')

      print('Begining heatmap to images')
      basecommand = 'python heatmap-to-image-casedirs.py ' + input_group_path + ' ' + output_group_path  # +' True'  #True only for 5X
      ####os.system(basecommand)
      print('\n')

      print('Begining compute results  ----4 patches')
      basecommand='python compare-predict-real-cluster_4x4.py '+output_group_path+' '+input_real_file
      os.system(basecommand)
      print('\n')

      print('Beginning compute results   ----3 patches')
      basecommand = 'python compare-predict-real-cluster_3x3.py ' + output_group_path + ' ' + input_real_file
      os.system(basecommand)
      print('\n')

      print('Beginning compute results   ----2 patches')
      basecommand = 'python compare-predict-real-cluster_2x2.py ' + output_group_path + ' ' + input_real_file
      os.system(basecommand)
      print('\n')

      print('Beginning compute results   ----1 patches')
      basecommand = 'python compare-predict-real-cluster_1x1.py ' + output_group_path + ' ' + input_real_file
      os.system(basecommand)
      print('\n')


def compute_statistics(posy, posy_scores):
       pos_count = 0
       neg_count = 0
       wrong_pos_count = 0
       wrong_neg_count = 0

       for i in range(len(posy)):
           if posy[i] == 1:  # pos case
               pos_count += 1

               if posy_scores[i] < 0.5:
                   wrong_pos_count += 1
           else:
               neg_count += 1

               if posy_scores[i] > 0.5:
                   wrong_neg_count += 1

       accuracy = float(pos_count + neg_count - wrong_pos_count - wrong_neg_count) / (pos_count + neg_count)
       sensitivity = (pos_count - wrong_pos_count + 1e-7) / (pos_count + 1e-7)
       specificity = (neg_count - wrong_neg_count + 1e-7) / (neg_count + 1e-7)

       if pos_count == 0 or neg_count == 0:
           auc = -1
       else:
           auc = metrics.roc_auc_score(posy, posy_scores)

       return accuracy, sensitivity, specificity, auc, pos_count, wrong_pos_count, neg_count, wrong_neg_count


def write_results(fw, accuracy, sensitivity, specificity, auc, posy, posy_scores, pos_count, wrong_pos_count, neg_count,
                  wrong_neg_count, result_name):
    fw.writelines('begin *******' + result_name + ' results ********** \n')
    fw.writelines('pos count: %d, wrong pos count: %d; neg count: %d, wrong neg count: %d\n' % (
    pos_count, wrong_pos_count, neg_count, wrong_neg_count))
    fw.writelines('accuracy, sensitivity, specificity: %.4f, %.4f, %.4f\n' % (accuracy, sensitivity, specificity))
    fw.writelines('AUC: %.4f\n' % (auc))
    fw.writelines('real label: ' + ''.join([str(i) + ' ' for i in posy]) + '\n')
    fw.writelines('predict value: ' + ''.join([str(i) + ' ' for i in posy_scores]) + '\n')
    fw.writelines('end ********' + result_name + ' results ********** \n\n')


def write_results_no_details(fw, accuracy, sensitivity, specificity, auc, posy, posy_scores, pos_count, wrong_pos_count,
                             neg_count,
                             wrong_neg_count, result_name):
    fw.writelines('begin *******' + result_name + ' results ********** \n')
    fw.writelines('pos count: %d, wrong pos count: %d; neg count: %d, wrong neg count: %d\n' % (
        pos_count, wrong_pos_count, neg_count, wrong_neg_count))
    fw.writelines('accuracy, sensitivity, specificity: %.4f, %.4f, %.4f\n' % (accuracy, sensitivity, specificity))
    fw.writelines('AUC: %.4f\n' % (auc))
    fw.writelines('end ********' + result_name + ' results ********** \n\n')


def create_statistics(output_results_path,output_base_path,output_results_tag,group_dirs):
   fw=open(output_results_path+output_results_tag+'_case_results','w')
   fw_no_details=open(output_results_path+output_results_tag+'_case_results_no_details','w')

   real_4P=''
   pre_4P=''
   real_3P=''
   pre_3P=''
   real_2P=''
   pre_2P=''
   real_1P=''
   pre_1P=''

   for group in group_dirs:
      lines = open(output_base_path + group + '/heatmap/statistics-by-cluster_4x4').readlines()  # get 4 patches results
      real_4P=real_4P+lines[1].strip().split(':')[1]
      pre_4P=pre_4P+lines[2].strip().split(':')[1]

      lines = open(output_base_path + group + '/heatmap_3x3/statistics-by-cluster_3x3').readlines()  # get 3 patches results
      real_3P=real_3P+lines[1].strip().split(':')[1]
      pre_3P=pre_3P+lines[2].strip().split(':')[1]

      lines = open(output_base_path + group + '/heatmap_3x3/statistics-by-cluster_2x2').readlines()  # get 3 patches results
      real_2P = real_2P + lines[1].strip().split(':')[1]
      pre_2P = pre_2P + lines[2].strip().split(':')[1]

      lines = open(output_base_path + group + '/heatmap_3x3/statistics-by-cluster_1x1').readlines()  # get 3 patches results
      real_1P = real_1P + lines[1].strip().split(':')[1]
      pre_1P = pre_1P + lines[2].strip().split(':')[1]

   real_4P_data = [int(i.strip()) for i in real_4P.strip().split()]
   pre_4P_data = [float(i.strip()) for i in pre_4P.strip().split()]

   real_3P_data = [int(i.strip()) for i in real_3P.strip().split()]
   pre_3P_data = [float(i.strip()) for i in pre_3P.strip().split()]

   real_2P_data = [int(i.strip()) for i in real_2P.strip().split()]
   pre_2P_data = [float(i.strip()) for i in pre_2P.strip().split()]

   real_1P_data = [int(i.strip()) for i in real_1P.strip().split()]
   pre_1P_data = [float(i.strip()) for i in pre_1P.strip().split()]

   fw.writelines('This is ' + output_results_tag + ' statistics results\n')
   fw_no_details.writelines('This is ' + output_results_tag + ' statistics results\n')

   accuracy, sensitivity, specificity, auc, pos_count, wrong_pos_count, neg_count, wrong_neg_count = compute_statistics(
       real_4P_data, pre_4P_data)
   write_results(fw, accuracy, sensitivity, specificity, auc, real_4P_data, pre_4P_data, pos_count, wrong_pos_count,
                 neg_count, wrong_neg_count, '4 patches')
   write_results_no_details(fw_no_details, accuracy, sensitivity, specificity, auc, real_4P_data, pre_4P_data,
                            pos_count, wrong_pos_count, neg_count, wrong_neg_count, '4 patches')

   accuracy, sensitivity, specificity, auc, pos_count, wrong_pos_count, neg_count, wrong_neg_count = compute_statistics(
       real_3P_data, pre_3P_data)
   write_results(fw, accuracy, sensitivity, specificity, auc, real_3P_data, pre_3P_data, pos_count, wrong_pos_count,
                 neg_count, wrong_neg_count, '3 patches')
   write_results_no_details(fw_no_details, accuracy, sensitivity, specificity, auc, real_3P_data, pre_3P_data,
                            pos_count, wrong_pos_count, neg_count, wrong_neg_count, '3 patches')

   accuracy, sensitivity, specificity, auc, pos_count, wrong_pos_count, neg_count, wrong_neg_count = compute_statistics(
       real_2P_data, pre_2P_data)
   write_results(fw, accuracy, sensitivity, specificity, auc, real_2P_data, pre_2P_data, pos_count, wrong_pos_count,
                 neg_count, wrong_neg_count, '2 patches')
   write_results_no_details(fw_no_details, accuracy, sensitivity, specificity, auc, real_2P_data, pre_2P_data,
                            pos_count, wrong_pos_count, neg_count, wrong_neg_count, '2 patches')

   accuracy, sensitivity, specificity, auc, pos_count, wrong_pos_count, neg_count, wrong_neg_count = compute_statistics(
       real_1P_data, pre_1P_data)
   write_results(fw, accuracy, sensitivity, specificity, auc, real_1P_data, pre_1P_data, pos_count, wrong_pos_count,
                 neg_count, wrong_neg_count, '1 patches')
   write_results_no_details(fw_no_details, accuracy, sensitivity, specificity, auc, real_1P_data, pre_1P_data,
                            pos_count, wrong_pos_count, neg_count, wrong_neg_count, '1 patches')

   fw.close()
   fw_no_details.close()


if statistics_method == 'EACH':
    for index in range(len(group_dirs)):
        group = []
        result_tag=output_results_tag[index]
        group.append(group_dirs[index])

        create_statistics(output_results_path, output_base_path, result_tag, group)

if statistics_method=='ALL':
    create_statistics(output_results_path, output_base_path, output_results_tag, group_dirs)