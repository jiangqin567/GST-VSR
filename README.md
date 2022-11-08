
# GUIDED SPATIO-TEMPORAL LEARNING METHOD FOR 4K VIDEO SUPER-RESOLUTION
The dataset can be download at link: https://pan.baidu.com/s/1NASPgxmLwirq98vm2eNIUA  fetch code: xhgq
# Testing #
First create the testing dataset.

## Creating the test set ## 

1. Put the downloaded test and train videos in the dollowing path:

`data/raw_videos`

For Pixabay-Set dataset, run the dollowing command:

`python data_process/generate_frame_from_video.py`

Then, please run the dollowing command to obtain the LR frame

'python data_process/generate_data_from_frame1.py'

## Start testing ## 

Run the following command:

`python test.py`
