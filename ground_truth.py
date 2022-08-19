import pandas as pd
import counter

# gt = ground truth
gt_csv = "ground_truth/ground_truth.csv"
gt_df = pd.read_csv(gt_csv)

for i in range(len(gt_df)):
    video_in = "ground_truth/gt_in/" + gt_df.loc[i, "file name"]
    video_out = "ground_truth/gt_out/" + gt_df.loc[i, "file name"]
    gt_df.loc[i, 'counted'] = counter.main(video_in, video_out)
    error_rate = (abs(gt_df.loc[i, 'counted'] - gt_df.loc[i, 'count'])) / ((gt_df.loc[i, 'count']) * 100)
    gt_df.loc[i, 'accuracy'] = 100 - error_rate

print(gt_df)

