import numpy as np
import torch
import joblib
from torcheval.metrics.functional import binary_accuracy, binary_confusion_matrix, binary_precision, binary_recall
from ForestDiffusion import ForestDiffusionModel as ForestFlowModel

from train import get_samples_labels

import yaml

def get_input_variables():
    # Load config from YAML file
    with open('test_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Use values from the config
    local_path = config['local_path']
    inpath = config['inpath']
    train_split = config['train_split']
    test_split = config['test_split']
    model_name = local_path + inpath + config['model_name']
    test_catalog = config['test_catalog']

    return model_name, test_catalog


""" NOTE. Old method to set the input variables
def get_input_variables():
    local_path = '/home/s.fiscale/conda/Models/forest_diffusion/'
    inpath = 'trained_models/'
    train_split = '90'
    test_split = '100'
    model_name = local_path + inpath + 'forest_kepler_q1-q17_dr24_split'+train_split+'_nt25_k10_nb70.pkl'
    test_set_name = 'tess_exofop_spoc-qlp_split'+test_split
    test_catalog = local_path + 'data/' + test_set_name + '.csv'

    return model_name, test_catalog
"""

def print_metrics(samples, labels, forest_model):
    labels = labels.astype(int)

    c_lab = forest_model.predict(samples, n_t=25, n_z=8)

    t_lab = torch.tensor(c_lab)
    t_star = torch.tensor(labels)

    accuracy = binary_accuracy(t_lab, t_star)
    precision = binary_precision(t_lab, t_star)
    recall = binary_recall(t_lab, t_star)
    conf_mat = binary_confusion_matrix(t_lab, t_star)#, normalize="pred")

    print("Accuracy: ", accuracy.item())
    print("Precision: ", precision.item())
    print("Recall: ", recall.item())
    print("Confusion matrix:\n", conf_mat)


if __name__ == '__main__':
    # Get input variables
    model_name, test_catalog = get_input_variables()
    # Load the model
    print(f"Loading model {model_name}\n")
    forest_model = joblib.load(model_name)
    # Test the model on the input dataset
    print(f"Testing on {test_catalog}\n")
    samples, labels = get_samples_labels(test_catalog)
    
    #samples = np.concatenate((samples, samples1)) se vuoi campioni da differenti dataset
    print_metrics(samples, labels, forest_model)
    del samples, labels