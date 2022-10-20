# FishNet - Detecting Invasive Species in the Mediterranean Sea
**FishNet is a neural-network model purposed to detect and classify fish in videos 
taken in the Mediterranean sea.**


### Preparing the Data:

In order to create the dataset for the training, the following command should be run:

    python3 data_preparation.py --data-dir=DATA_DIR

The original full videos should be placed under `DATA_DIR` and this value is mandatory for the preparation process.

**Additional options:**

`--frames-dir` - The tagged frames are saved under the `frames_dir` directory, default value is 'frames'.

`--is-converted` - specify if the videos are original HD format, or converted (compressed) videos

`--create-frames` and `--create-bg-frames` - indicates whether to create the frames (or the background frames).

`--data-types` - Choose between using only the fish labels (2), only the background (1) or using both (0), default is 0.

`--create-augs` - Create augmentations for the frames, default is False.

`--num-augs` - Number of augmentation to create for each frame, default is 10.


Alternatively, this could be done by calling `prepare_data` function in `data_preparation.py` file.

### Training the Model:

To train the model, run the following:

    python3 fish_net.py --frames-dir=FRAMES_DIR

`FRAMES_DIR` is the directory in which the frames from the data-preparation stage are saved. 
If the data preparation stage was not executed, it will be executed at this stage, and the videos should 
be under `DATA_DIR` location. 

**Additional options:**

`--data-dir` - The location of the raw videos, in case there is a need to create the frames.

`--output-file` -  Specify model file name. If not specified, model will not be saved, default=None.

`--is-converted` - if True, use converted (compressed) videos instead of HD videos.

`--data-types` - Choose between using only the  fish labels (2), only the background (1) or using both (0), default is 0.

`--use-augs` - Use (and create) augmentations in the training process.

`--num-augs` - Number of augmentation to create for each frame, default is 10.

Alternatively, the model can be trained by calling `train_fishnet` function in `fish_net.py` file.

### Testing the Model:

This functionality cannot be executed via command line, and must be executed using the function `test_model` in `test_model.py`.

** function arguments**

`video_paths` - list - The video names on which to test the model.
`model_path` - str - model file name
`frames_dir` - str - frames directory

### Notebook example:

A running example to the different stages using python can be found in `FishNet.ipynb` file. 

