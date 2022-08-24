import pandas as pd
import counter

# gt = ground truth
gt_csv = "ground_truth/ground_truth.csv"
gt_df = pd.read_csv(gt_csv, na_values=0)

for i in range(len(gt_df)):
    video_in = "ground_truth/gt_in/" + gt_df.loc[i, "file name"]
    video_out = "ground_truth/gt_out/" + gt_df.loc[i, "file name"]
    gt_df.loc[i, 'counted'] = counter.main(video_in, video_out)
    accuracy = abs(gt_df.loc[i, 'counted'] - gt_df.loc[i, 'count'])
    accuracy = accuracy / gt_df.loc[i, 'count']
    accuracy = accuracy * 100
    accuracy = 100 - accuracy
    gt_df.loc[i, 'accuracy'] = round(accuracy, 2)

print(gt_df)
gt_df.to_csv(path_or_buf="ground_truth/gt_out/gt_out.csv", index=False)
