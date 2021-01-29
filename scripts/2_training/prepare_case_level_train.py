input_file=['/media/disk4/pathology_data_process/group17/heatmap_using_group_neg_V3/case_level_feature',\
              '/media/disk4/pathology_data_process/group16/heatmap_using_group_neg_V3/case_level_feature']
input_real_file='/media/disk4/pathology_data_process/xiangya-real.txt'
output_train_data='/media/disk4/pathology_data_process/train_data.txt'

case_data=[]
labels=[]
for one_file  in input_file:
    temp_data=open(one_file).readlines()

    for line in temp_data:
       case_data.append(line)

real_lines=open(input_real_file).readlines()
real_ids_labels={}
for line in real_lines:
    id,label=line.strip().split()
    real_ids_labels[id]=label             #read ids and labels

train_data=[]
all_ids=[]
for line in case_data:
    parts = line.strip().split()     #get features
    feature_5X = parts[-1].strip(',')
    feature_7X = parts[-2].strip(',')
    feature_10X = parts[-3].strip(',')

    id = line[0:7].strip()  # try to get id
    if real_ids_labels.has_key(id):
        temp_data='%s %s  %s  %s    %s'%(id,feature_10X,feature_7X,feature_5X,real_ids_labels[id])
        train_data.append(temp_data)
        all_ids.append(id)
        continue

    #for old id format
    if  len(id)==6:
        id='0'+id     #expand 6 to 7 bit id, new id format
    elif len(id)==7 and id[0]=='0':
        id=id[1:-1]    #shrink 7 to 6 bit id, old id format because some files use old format

    if real_ids_labels.has_key(id):
        temp_data='%s %s  %s  %s    %s'%(id,feature_10X,feature_7X,feature_5X,real_ids_labels[id])
        train_data.append(temp_data)
        all_ids.append(id)
        continue

    if len(id)==7:
        id=id[0:6]

    if real_ids_labels.has_key(id):
        temp_data='%s %s  %s  %s    %s'%(id,feature_10X,feature_7X,feature_5X,real_ids_labels[id])
        train_data.append(temp_data)
        all_ids.append(id)
        continue

    id='0'+id
    if real_ids_labels.has_key(id):
        temp_data='%s %s  %s  %s    %s'%(id,feature_10X,feature_7X,feature_5X,real_ids_labels[id])
        train_data.append(temp_data)
        all_ids.append(id)
        continue

    print('find a wrong id:'+id)

#remove same id cases with many slides
all_ids=list(set(all_ids))
unique_data=[]

for one_id in all_ids:
    max_norm=-1
    feature_7X='0'
    feature_10X='0'
    feature_5X='0'
    label='0'

    for line  in train_data:
        id, T_feature_10X, T_feature_7X, T_feature_5X, T_label=line.strip().split()

        if  id!=one_id:
            continue

        cur_norm = float(T_feature_5X) ** 2 + float(T_feature_7X) ** 2 + float(T_feature_10X) ** 2

        if cur_norm>max_norm:
            max_norm = cur_norm
            feature_5X = T_feature_5X
            feature_7X = T_feature_7X
            feature_10X = T_feature_10X
            label=T_label

    temp_data = '%s %s  %s  %s    %s' % (one_id, feature_10X, feature_7X, feature_5X, label)
    unique_data.append(temp_data)

fw=open(output_train_data,'w')
for line in unique_data:
    fw.writelines(line + '\n')

fw.close()



