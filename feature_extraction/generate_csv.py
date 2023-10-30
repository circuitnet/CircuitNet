import os
import csv
import random

feature_path = './training_set/sample/feature'
label_path = './training_set/sample/label'

out_train_csv = './train.csv'
out_test_csv = './test.csv'

split_ratio = 0.7  # 0.7 train 0.3 test

if __name__ == '__main__':
    random.seed(230907)
    features = os.listdir(feature_path)
    labels = os.listdir(label_path)
    
    features = [v for v in features]
    labels = [ v for v in labels]

    assert len(features) == len(labels)

    with open(out_train_csv, 'w') as f_train:
        with open(out_test_csv, 'w') as f_test:
            f_train_csv = csv.writer(f_train, delimiter=',')
            f_test_csv = csv.writer(f_test, delimiter=',')

            for i, features_name in enumerate(features):
                features_path = 'training_set/sample/feature/{}'.format(features_name)
                labels_path = 'training_set/sample/label/{}'.format(features_name)
                instance_count_path = 'out/features/instance_count/{}'.format(features_name)
                instance_IR_drop_path = 'out/features/instance_IR_drop/{}'.format(features_name)
                instance_name_path = 'out/features/instance_name/{}z'.format(features_name[:-1])
                if len(features) == 2:
                    if i == 0: 
                        f_train_csv.writerow([features_path, labels_path])
                        f_test_csv.writerow([features_path, labels_path, instance_count_path, instance_IR_drop_path, instance_name_path])
                        print('train: {}'.format(i))
                    else:
                        f_test_csv.writerow([features_path, labels_path, instance_count_path, instance_IR_drop_path, instance_name_path])
                        print('test: {}'.format(i))
                else:
                    if random.random() <= split_ratio:
                        f_train_csv.writerow([features_path, labels_path])
                        print('train: {}'.format(i))
                    else:
                        f_test_csv.writerow([features_path, labels_path, instance_count_path, instance_IR_drop_path, instance_name_path])
                        print('test: {}'.format(i))
