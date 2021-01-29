#coding:utf-8
import sys
from sklearn import metrics

if len(sys.argv)>=3:
    base_path=sys.argv[1]
    input_real_file=sys.argv[2]
else:
    base_path = '/media/disk6/pathology_data_process/group3/'
    input_real_file='/media/disk4/pathology_data_process/xiangya-real.txt'
 #   input_real_file = '/media/disk4/xiangya-3/xiangya-3-real.txt'

input_predict_file=base_path+'heatmap_3x3/cluster_value_3x3'
output_file=base_path+'heatmap_3x3/predict-real-by-cluster_1x1'
output_statistics_file=base_path+'heatmap_3x3/statistics-by-cluster_1x1'     #statistics results

fw1=open(input_real_file)
fw2=open(input_predict_file)
fw3=open(output_file,'w')
fw4=open(output_statistics_file,'w')
threshold=1

real_lines=fw1.readlines()
predict_lines=fw2.readlines()

for line in real_lines:
    id, label= line.strip().split()
    flag=0
    info=[]

    for line2 in reversed(predict_lines):
        may_id=line2.strip()[0:len(id)]

        if may_id!=id:
            continue
        else:
            parts=line2.split()
            cluster_patches=parts[-1].strip(',')
            total_patches=parts[-2].strip(',')
            pos_patches=parts[-3].strip(',')

            info.append('%s %6s %8s   %8d   %s'%(id,pos_patches,total_patches,int(pos_patches),label))

            predict_lines.remove(line2)
            flag=1

    if flag==1:
        max_cent=-1
        total_patches=''
        pos_patches=''

        if len(info)>1:
            a=0

        for one_info in info:
            parts = one_info.split()
            cluster_patches = parts[3]
            label=parts[4]

            if int(cluster_patches)>max_cent:
                max_cent=float(cluster_patches)
                total_patches = parts[2]
                pos_patches = parts[1]

        one_info='%s %6s %8s   %8d   %s' % (id, pos_patches, total_patches, int(max_cent), label)

        if float(max_cent) < threshold and label == '1':
            one_info = one_info + '     ?'

        if float(max_cent) >= threshold and label == '0':
            one_info = one_info + '     ?'

        fw3.writelines(one_info + '\n')

  #  if flag==0:
   #     print('id '+id+' has no predict id')

#for 6 old pathlogy bit, it may be expand 7(add 0) or still be 6 bit
for line in real_lines:
    id, label= line.strip().split()
    flag=0
    info=[]

    if id[0]=='1':  #id with 1 begining is neglected, it is new id format
        continue

    #for old id format
    if  len(id)==6:
        id='0'+id     #expand 6 to 7 bit id, new id format
    elif len(id)==7 and id[0]=='0':
        id=id[1:]    #shrink 7 to 6 bit id, old id format because some files use old format

    for line2 in reversed(predict_lines):
        may_id=line2.strip()[0:len(id)]

        if may_id!=id:
            continue
        else:
            parts=line2.split()
            cluster_patches=parts[-1].strip(',')
            total_patches=parts[-2].strip(',')
            pos_patches=parts[-3].strip(',')

            info.append('%s %6s %8s   %8d   %s'%(id,pos_patches,total_patches,int(pos_patches),label))

            predict_lines.remove(line2)
            flag=1

    if flag==1:
        max_cent=-1
        total_patches=''
        pos_patches=''

        if len(info)>1:
            a=0

        for one_info in info:
            parts = one_info.split()
            cluster_patches = parts[3]
            label=parts[4]

            if int(cluster_patches)>max_cent:
                max_cent=float(cluster_patches)
                total_patches = parts[2]
                pos_patches = parts[1]

        one_info='%s %6s %8s   %8d   %s' % (id, pos_patches, total_patches, int(max_cent), label)

        if float(max_cent) < threshold and label == '1':
            one_info = one_info + '     ?'

        if float(max_cent) >= threshold and label == '0':
            one_info = one_info + '     ?'

        fw3.writelines(one_info + '\n')

 #   if flag==0:
  #      print('id '+id+' has no predict id')


if len(predict_lines)!=0:
    #print('some predict has no real id')
    fw3.writelines('\n'+'no label sample, and begin guess it is positive .....\n')

    for line in predict_lines:
        #print(line)
        parts = line.split()
        cluster_patches = parts[-1].strip(',')
        total_patches = parts[-2].strip(',')
        pos_patches = parts[-3].strip(',')
        id=parts[0]

        one_info='%s %6s %8s   %8d   %s'%(id,pos_patches,total_patches,int(pos_patches),str(1))

        if float(cluster_patches) < threshold:
            one_info = one_info + '     ?'

        fw3.writelines(one_info + '\n')

fw1.close()
fw2.close()
fw3.close()

lines=fw3=open(output_file).readlines()

pos_count=0
neg_count=0
wrong_pos_count=0
wrong_neg_count=0
posy=[]
posy_scores=[]
for line in lines:
    parts=line.strip().split()

    if len(parts)<5 or parts[0]=='no':
        break

    real_label= int(parts[4])
    if real_label>1:
        real_label=1

    posy.append(real_label)        #real_label

    if real_label==0:
       if int(parts[1])>=threshold:     #wrong neg sample
           posy_scores.append(1.0)
           wrong_neg_count += 1
           neg_count += 1
       else:
           posy_scores.append(0.0)
           neg_count+=1
    else:
        if int(parts[1]) < threshold:  # wrong neg sample
           posy_scores.append(0.0)
           wrong_pos_count += 1
           pos_count += 1
        else:
           posy_scores.append(1.0)
           pos_count += 1

#comput accuracy and sensitivity and so on
accruray=float(pos_count+neg_count-wrong_pos_count-wrong_neg_count)/(pos_count+neg_count)
sensitivity=(pos_count-wrong_pos_count+1e-7)/(pos_count+1e-7)
specificity=(neg_count-wrong_neg_count+1e-7)/(neg_count+1e-7)
print('accuracy, sensitivity, specificity are:  %.4f, %.4f, %.4f\n'%(accruray,sensitivity,specificity))
fw4.writelines('accuracy, sensitivity, specificity are:  %.4f, %.4f, %.4f\n'%(accruray,sensitivity,specificity))

print(posy)
print(posy_scores)
fw4.writelines('real label: '+''.join([str(i)+' ' for i in posy])+'\n')
fw4.writelines('predict value: '+''.join([str(i)+' ' for i in posy_scores])+'\n')

#compute AUC
if pos_count==0 or neg_count==0:
   exit()

test_auc = metrics.roc_auc_score(posy,posy_scores)
print('AUC is : %.4f'%(test_auc))

fw4.writelines('AUC is: %.4f\n'%(test_auc))
fw4.close()




