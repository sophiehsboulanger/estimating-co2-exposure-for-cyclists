import time

import pandas as pd
import counter

# gt = ground truth
gt_csv = "ground_truth/ground_truth_mini.csv"
gt_df = pd.read_csv(gt_csv, na_values=0)

for i in range(len(gt_df)):
    video_in = "ground_truth/gt_in/" + gt_df.loc[i, "file"]
    print(video_in)
    video_out = "ground_truth/gt_out/" + gt_df.loc[i, "file"]
    start_time = time.time()
    results = counter.main(video_in, video_out,frame_skip=6, max_age=30,nms_max_overlap=1, min_age=3, min_size=0.4)
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    print(elapsed_time)
    gt_df.loc[i, 'counted'] = results['total count']
    # gt_df.loc[i, 'car'] = results['cars']
    # gt_df.loc[i, 'bus'] = results['buses']
    # gt_df.loc[i, 'truck'] = results['trucks']
    # gt_df.loc[i, 'motorbike'] = results['motorbikes']
    #
    accuracy = abs(gt_df.loc[i, 'counted'] - gt_df.loc[i, 'true count'])
    accuracy = accuracy / gt_df.loc[i, 'true count']
    accuracy = accuracy * 100
    accuracy = 100 - accuracy
    gt_df.loc[i, 'accuracy'] = round(accuracy, 2)

print(gt_df)
gt_df.to_csv(path_or_buf="ground_truth/gt_out/ground_truth_mini.csv", index=False)

