import pandas as pd
'''This script simply takes the large annotation file and splits it into individual files per image
'''
annotations = pd.read_csv('inputs/detector_test/_annotations.csv', names=['file', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
annotations = annotations[['file', 'xmax', 'ymax', 'xmin', 'ymin', 'class']]
grouped = annotations.groupby(['file'])
for file in grouped:
    file_name = file[0][:-4]
    file[1].to_csv('inputs/detector_test/labels/{}.csv'.format(file_name), index=False)