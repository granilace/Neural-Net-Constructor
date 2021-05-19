import os
import pandas as pd
import random
import shutil

IMGS_DIR = 'data/cifar-10/raw/train'
LABELS_PATH = 'data/cifar-10/raw/trainLabels.csv'

labels_df = pd.read_csv(LABELS_PATH)
labels = list(labels_df['label'].unique())
labels_df['label'] = labels_df['label'].apply(lambda label: labels.index(label))
labels_df['fname'] = labels_df['id'].apply(lambda file_id: '{}.png'.format(file_id))
labels_df.drop(['id'], axis=1).to_csv('data/cifar-10/prepared/labels.csv', index=False)

for fname in os.listdir(IMGS_DIR):
    src_path = os.path.join(IMGS_DIR, fname)
    if not os.path.isfile(src_path):
        continue
    dst_path = os.path.join('data/cifar-10/prepared', 'train' if random.random() < 0.8 else 'test', fname)
    shutil.copyfile(src_path, dst_path)

