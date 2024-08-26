# README #

## Installation and set up
Required packages are found in `env.yaml`, the repo was written using 3.10.13 on Windows 11.

### Downloading dataset
Download the dataset from [here] Note: dataset is still under review [here](https://data.4tu.nl/), the correct link to download RIFTSeg will be inserted once the dataset is published. 

Unpack the files into a `data` folder. This should have the following structure:
```
data
|-- annotations
    |-- all_annotations.json
    |-- split_mapping.json
|-- images
    |-- loc1
    |-- loc2
    ....
    |-- loc6
|-- pretrained_yolo_models
    |-- different_train_sizes
    |-- generalization_loc6
    |-- trained_one_location
    |-- best_model.pt
|-- timeseries
    |-- ts_loc1_1
    |-- ts_loc1_2
    |-- ts_loc5_1
```
Note that the filenames in `split_mapping.json` are formatted as `<location_id>_filename.jpg`.

The `pretrained_yolo_models` folder contains the YOLO models used in the paper. This folder should be moved to the `models` folder. 
The `timeseries` folder contains 3 subfolder, each containing a set of unannotated timeseries data. In the `notebook.ipynb`, there is an example cell showing how this timeseries data can be used to show consistency of predictions.

### Preparing data
To prepare the data for model training and evaluation on the dataset, run the cells in the `prepare_data.ipynb` notebook in the notebooks folder. Afterwards, the `data` folder should have the following structure:

```
data
|-- annotations
    |-- split_<trainsize>_<testsize>
        |-- test
            |-- images
                |-- <imageid>.jpg
            |-- labels
                |-- <imageid>.txt
            |-- annotations.json
        |-- train
    |-- all_annotations.json
    |-- split_mapping.json
|-- combined_gt_masks
    |-- loc1.pt
    |-- loc2.pt
    ...
    |-- loc6.pt
|-- images
    |-- loc1
    |-- loc2
    ....
    |-- loc6
|-- timeseries
    |-- ts_loc1_1
    |-- ts_loc1_2
    |-- ts_loc5_1
```
The `all_annotations.json` contains annotations for the entire dataset, and is used in all methods contained in the `main.py` file. This includes evaluation of YOLO models, prompting of SegGPT and PerSAM, and visualization of ground truth masks. 

The `annotations.json` files within the splits are used for training of YOLO models only. 

Care was taken to ensure that using different splits for training and evaluation does not cause data leakage. The 20% test set is the smallest, and each larger test set is a superset of that set. This allows for a model trained on 60% of the dataset to be evaluated safely on the 20% test split, since no datapoints used in training that model are included in the test set.

## Predicting masks
With `main.py` you can predict segmentation masks on (a subset of) the dataset. Each model has class defining specific prediction and preprocessing methods for that model. The full predictions are handled by the `SegmentationModel` class. 

### Arguments
#### General
- `annotation_path`: Path to a .json file containing COCO style annotations
- `images_path` : Path to the dataset folder. Default: `./data/images`
- `n_to_visualize` : Set to non-zero to visualize batches of predictions. Batch size for visualization is set to 5. 
- `timeseries`: Used to predict on data without ground truth annotations, for example unlabeled timeseries data. This bypasses the visualization and metric computation steps. Argument takes the path to the folder containing relevant data.

#### Experiments
- `model`: Dictates which model to use, choices are 'PerSAM', 'SegGPT', 'RandomForest' and 'YOLO'.
- `location`: Which location to use, number between 1 and 6.
- `remove_posthoc`: Enable flag to remove irrelevant masks after prediction. These are masks that are outside of the area of interest (the river).
- `prompt_imgs`: Dictates which images are used as prompts. Enter as list of ints, i.e. `--prompt_imgs [1,2,3]`.
- `match_prompt_img`: Use flag to find the best prompt image for each testing image, using image embeddings.
- `n_patches`: Set to >1 to divide image into patches to predict on. Setting to `n` will create `n^2` patches.

#### SegGPT
- `binary_mask_cutoff`: Dictates the cutoff for the logit output of SegGPT to convert to a binary mask.

#### PerSAM
- `min_feature_sim`: When mean feature similarity drops below this value, prediction of masks stops.
- `persam_training`: Use flag to use PerSAM-F instead of PerSAM.

#### YOLO
- `yolo_test_path`: Use this argument to override the `images_path` folder and select images from a split used for YOLO training and testing. 
- `yolo_model_path`: Path to `.pt` file containing YOLO Segmentation model.

## Training YOLO
You can train YOLO models with `yolo_training/train_yolo.py` and control which locations are included in training, as well as which checkpoint is loaded as a base model. When excluding one or multiple locations from training, a new folder will be created in `data/annotations` containing the relevant split for only the selected locations.
### Arguments
- `split_folder`: Path to folder containing splits
- `train_size`: Determine size of split to train on
- `experiment_name`: Allows easier identification of training runs
- `locations`: Dictates which locations will be used for training, enter as string of ints seperated by a space.
- `model_version`: Path to `.pt` file containing (pre-trained) YOLOv8 Segmentation model
- `batch_size`: Batch size used for training
- `img_size`: Image sized used for training
- `epochs`: Number of epochs for training
