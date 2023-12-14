statement of the list files

| dataset name     | List description  |  
| ------------- |-------------| 
| `ACDC`     | `train.list` case name of all the training dataset |  DeepLabv3+ works better and faster  |
| `num_labels`     | number of labelled images in the training set, choose `0` for training all labelled images  | only available in the full label mode  |
| `partial`     |  percentage of labeled pixels for each class in the training set, choose `p0, p1, p5, p25` for training 1, 1%, 5%, 25% labelled pixel(s) respectively  | only available in the partial label mode |
| `num_negatives` | number of negative keys sampled for each class in each mini-batch | only applied when training with ReCo loss|