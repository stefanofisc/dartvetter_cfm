import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_input_variables():
    # Input data
    precision = [0.6729435324668884, 0.6633639335632324, 0.6574478149414062, 0.6756756901741028]
    recall = [0.9430555701255798, 0.953125, 0.9409722089767456, 0.9375]
    accuracy = [0.8081894516944885, 0.802532970905304, 0.7957128286361694, 0.8096957206726074]
    labels = ['split 50-50', 'split 60-40', 'split 80-20', 'split 90-10']

    output_path = '/home/s.fiscale/conda/Models/forest_diffusion/output_files/plot/'
    output_file = output_path + 'train_test_kepler_dr24.png'

    return precision, recall, accuracy, labels, output_file


def plot_confmat(confusion_matrix, label_matrix, split, output_file):
    # Create the heatmap with larger text size in the cells
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix, annot=label_matrix, fmt='', cmap='Blues', cbar=False, square=True, linewidths=1, linecolor='black', annot_kws={"size": 16})

    # Add title and labels
    plt.title('Confusion Matrix - Split ' + split, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    # Show plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=1200)



def plot_evaluation_metrics(precision, recall, accuracy, labels, output_file):
    # Set the style
    plt.style.use('seaborn-darkgrid')

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each metric with improved visuals
    ax.plot(labels, precision, label='Precision', marker='o', markersize=8, linestyle='-', color='blue', linewidth=2)
    ax.plot(labels, recall, label='Recall', marker='s', markersize=8, linestyle='--', color='green', linewidth=2)
    ax.plot(labels, accuracy, label='Accuracy', marker='^', markersize=8, linestyle='-.', color='red', linewidth=2)

    # Add title and labels with enhanced fonts
    ax.set_title('Model Metrics across Different Data Splits', fontsize=16, fontweight='bold')
    ax.set_xlabel('Data Splits', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)

    # Add legend with a frame and better positioning
    ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

    # Rotate x-axis labels and improve layout
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Show the plot
    plt.savefig(output_file, dpi=1200)


def main_confmat():
    train_split = '100'
    test_split = '100_dtt9'
    split = train_split+'-'+test_split
    # Confusion matrix values
    tn = 491
    fp = 36
    fn = 1262
    tp = 52
    confusion_matrix = np.array([[tn, fp],
        [ fn, tp]])
    # Labels for the heatmap
    label_matrix = np.array([[f'TN\n{tn}', f'FP\n{fp}'],[f'FN\n{fn}', f'TP\n{tp}']])
    output_path = '/home/s.fiscale/conda/Models/forest_diffusion/output_files/plot/'
    output_file = output_path + 'confmat_kepler_dr25_train'+train_split+'_test'+test_split+'.png'
    plot_confmat(confusion_matrix, label_matrix, split, output_file)

if __name__ == '__main__':
    main_confmat()
    #precision, recall, accuracy, labels, output_file = get_input_variables()
    #plot_evaluation_metrics(precision, recall, accuracy, labels, output_file)