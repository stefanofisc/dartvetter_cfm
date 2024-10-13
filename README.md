# dartvetter_cfm
Code and resources for training the Forest Diffusion model to classify potential planetary signals

## Project Overview
This repository contains the code and resources necessary to train and test the Forest Diffusion Model, a machine learning model designed to distinguish planetary signals from false positives. The model is compatible with signals from the Kepler and TESS missions, but can also work with data from any transiting survey. It utilizes global view data and features extracted from a Convolutional Neural Network (CNN) for this classification task.

## Setup and Dependencies
Create a new conda environment and set all the requested libraries to correctly run the code

```bash
conda create --name environment_name python=3.10
conda activate environment_name
```
Once the environment is activated, install the necessary libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
After installing the libraries, you can verify the installation by running the following command:
```bash
pip list
```
This will display all the libraries installed in your environment.


## Directory Structure
### Files
- `train.py`: This script contains methods for training the Forest Diffusion model. It loads data from the data/directory, processes it, and trains the model using the specified hyperparameters. After training, the trained model is saved in the trained_models/ directory.
- `test.py`: Contains methods for evaluating the model's performance. It loads a trained model from trained_models/ and runs tests on the test dataset.
- `plot.py`: Used to generate visualizations of the model's performance, including accuracy, loss curves, and confusion matrices.
train.slurm & test.slurm: SLURM batch files for running the training and testing processes on a cluster environment. These scripts are designed for distributed training on high-performance compute nodes.

### Directories
- `data/`
  
This directory contains input data used for training and testing the model. Input data can be in .csv or .npy format.
  - CSV Format: Each CSV file is a table of size NxM, where:
    - N = number of elements in the dataset
    - M = input feature size Each row represents a "global view" (length 201) and the associated label. Example: <feature1, feature2, ..., feature201, label>
Naming Convention: <mission>_<catalog_name>_<dataset_size>. Example: kepler_q1-q17_dr25_split90.csv.  Important: Ensure that training and test splits sum to 100% (e.g., [split90,split10] or [split80,split20]).

  - .npy Format: .npy files store K-dimensional vectors that contain the features extracted by the CNN for each global view. This is used for faster loading during training.

- `output_files/`
  
Stores the outputs generated during training and testing. This includes logs, evaluation metrics, and intermediate results.

- `trained_models/`
  
After training, models are saved in this directory. The naming convention is:
<model_type>_<mission>_<catalog_name>_<dataset_size>.pth

- `utils/`
  
Contains utility scripts.
-- `convert_tfrecord_to_numpy.py`: Converts input data from .tfrecord format to NumPy arrays. This step is necessary to process the data with the ForestDiffusion model.

- `feature_extraction/`
  
This directory contains the methods used to extract features from the input data using a CNN model.
-- `m1.py`: Defines the architecture of the CNN used for feature extraction.
-- `train_test_cnn.py`: Contains methods for training and testing the CNN. After training, the extracted features are saved as .npy files, which are later used by the Forest Diffusion model.


## Usage example

To train the Forest Diffusion model, run:

```bash
python train.py --config train_config.yaml
```

To test a trained model, run:
```bash
python test.py --config test_config.yaml
```

Or you can edit the `train(test).slurm` files in order your code to be run on a remote environment. Make sure your data is correctly placed in the `data/` directory and the configuration file contains the right parameters.
