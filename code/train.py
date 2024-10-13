import numpy as np
import torch
import joblib
from torcheval.metrics.functional import binary_accuracy, binary_confusion_matrix, binary_precision, binary_recall
from ForestDiffusion import ForestDiffusionModel as ForestFlowModel
from feature_extraction.read_extracted_features import get_features_labels, get_features_labels_filename

# Load input parameters from the configuration file
import yaml

def get_input_variables():
    # Open the YAML config file
    with open('train_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # Use values from the config
    local_path = config['local_path']
    input_catalog = config['input_catalog']
    output_dir = config['output_dir']
    model_name = config['model_name']
    n_t = config['n_t']
    k = config['k']
    n_batch = config['n_batch']
    train_model_on_global_views = config['train_model_on_global_views']

    return local_path, input_catalog, output_dir, model_name, n_t, k, n_batch, train_model_on_global_views



def get_samples_labels(path):
    values = np.genfromtxt(path, delimiter=',')
    samples = values[:,:-1]
    labels =  values[:,-1]

    return samples, labels


if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get input variables
    local_path, input_catalog, output_dir, model_name, n_t, k, n_batch, train_model_on_global_views = get_input_variables()
    # The model can be trained on global views or on the features extracted from the global views
    if train_model_on_global_views:
        samples, labels = get_samples_labels(local_path + input_catalog)
    else: 
        filename_features, filename_labels = get_features_labels_filename()
        samples, labels = get_features_labels(filename_features, filename_labels)
    labels = labels.astype(int)

    print(f"Size of train examples: {samples.shape} and {labels.shape}\n")
    # Move data to the device (GPU or CPU)
    samples = torch.tensor(samples).to(device)
    labels = torch.tensor(labels).to(device)

    print("Training")
    forest_model = ForestFlowModel(
            samples,
            label_y=labels,
            n_t=n_t,
            duplicate_K=k,
            bin_indexes=[],
            cat_indexes=[],
            int_indexes=[],
            diffusion_type="flow",
            n_jobs=-1,
            n_batch = n_batch,
            seed=1,
    )
    # Save the model
    joblib.dump(forest_model, output_dir + model_name)
