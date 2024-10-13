import numpy as np

def get_features_labels_filename():
    local_path = '/home/s.fiscale/conda/Models/forest_diffusion/data/'
    filename_features = local_path + 'kepler_q1-q17_dr25_split100_extracted_features.npy'
    filename_labels = local_path + 'kepler_q1-q17_dr25_split100_extracted_labels.npy'

    return filename_features, filename_labels

def get_features_labels(filename_features, filename_labels):
    # Carica il file .npy
    features = np.load(filename_features)
    labels = np.load(filename_labels)

    return features, labels # numpy.ndarray()


if __name__ == '__main__':
    print("read extracted features")
    # Esempio di utilizzo
    # Specifica il percorso del file .npy
    local_path = '/home/s.fiscale/conda/Models/forest_diffusion/data/'
    filename_features = local_path + 'kepler_q1-q17_dr25_split100_extracted_features.npy'
    filename_labels = local_path + 'kepler_q1-q17_dr25_split100_extracted_labels.npy'
    # Carica il file .npy
    features = np.load(filename_features)
    labels = np.load(filename_labels)

    # Stampa il contenuto del file
    print(features[0:16134].shape)
    print(labels[0:16134].shape)