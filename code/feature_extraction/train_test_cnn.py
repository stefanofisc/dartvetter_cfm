import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import m1 

def get_samples_labels(path):
    values = np.genfromtxt(path, delimiter=',')
    samples = values[:,:-1]
    labels =  values[:,-1]

    return samples, labels


def compute_class_weights(labels):
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    return torch.tensor(weight, dtype=torch.float)



# Funzione di addestramento con salvataggio del modello e delle caratteristiche estratte
def train_model(model, train_loader, criterion, optimizer, num_epochs=25, save_path="cnn_default.pt", save_features=False):
    if save_features:
        extracted_features = []  # Lista per memorizzare le caratteristiche estratte
        extracted_labels = []    # Lista per memorizzare i label associati alle caratteristiche estratte

    metric_tracker = m1.MetricTracker()  # Crea un'istanza di MetricTracker

    for epoch in range(num_epochs):
        model.train()  # Imposta il modello in modalità training
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Inizializzo i gradienti
            optimizer.zero_grad()
            
            # Feed-forward pass
            features = model.global_view(inputs, trainingval=True)      # Estrazione delle features dall'input
            outputs = model.fully_connected(features) # Classificazione delle feature estratte mediante livello fully-connected
            
            labels = labels.squeeze()  # Elimina eventuali dimensioni aggiuntive
            loss = criterion(outputs, labels.unsqueeze(1).float())
            # Backpropagation pass
            loss.backward()
            optimizer.step()

            # Operazioni da effettuare alla fine della i-sima epoca di training
            running_loss += loss.item() * inputs.size(0)    # Calcolo della loss 
            metric_tracker.update(outputs, labels, loss)    # Aggiorna le metriche
        
            # Salva le caratteristiche estratte solo durante l'ultima epoca
            if save_features:
                if epoch == num_epochs - 1:
                    extracted_features.append(features.detach().cpu().numpy())
                    extracted_labels.append(labels.detach().cpu().numpy())
            
        
        epoch_loss = running_loss / len(train_loader.dataset)
        # Log delle metriche al termine dell'epoca
        precision, recall, f1 = metric_tracker.compute()
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Loss: {epoch_loss:.4f}')

    
    # Salva il modello addestrato
    torch.save(model.state_dict(), save_path)
    print(f"Modello salvato in: {save_path}")
    
    # Salva le caratteristiche estratte in un file numpy (solo per l'ultima epoca)
    # Salva anche i labels associati ai campioni per cui stai salvando le features estratte
    
    if save_features: 
        if extracted_features:
            features_array = np.vstack(extracted_features)
            labels_array = np.concatenate(extracted_labels)
            np.save('extracted_features.npy', features_array)
            np.save('extracted_labels.npy', labels_array)
            print(f"Caratteristiche estratte salvate in 'extracted_features.npy'")
            print(f"Label associati salvati in 'extracted_labels.npy'")
    
    
    return model


def evaluate_model(model, test_loader, criterion):
    model.eval()  # Imposta il modello in modalità valutazione
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Disabilita il calcolo dei gradienti durante la valutazione
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            features = model.global_view(inputs, trainingval=False)  # Caratteristiche estratte da GlobalViewFiscale
            outputs = model.fully_connected(features)
            
            # Calcola la perdita
            loss = criterion(outputs, labels.unsqueeze(1))  # Adattato per output a singolo neurone
            running_loss += loss.item() * inputs.size(0)
            
            # Salva predizioni e etichette reali per il calcolo delle metriche
            preds = torch.round(torch.sigmoid(outputs))  # Approssima a 0 o 1 per le predizioni
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Calcola la loss media
    test_loss = running_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    
    # Concatena le predizioni e le etichette lungo il batch dimensionale
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calcolo delle metriche
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    
    return test_loss, precision, recall, f1, conf_matrix


if __name__ == '__main__':
    train_test_from_scratch = False
    # Iperparametri rete neurale
    rateval = 0.3       # Dropout probability rate
    learning_rate = 0.001  # Learning rate
    batch_size = 128     # Dimensione del batch
    num_epochs = 10     # Numero di epoche
    # Path di training e test set
    local_path = '/path/to/dartvetter_cfm/'
    training_set_path = local_path + 'data/kepler_q1-q17_dr24_split90.csv'
    test_set_path = local_path + 'data/tess_exofop_spoc-qlp_split100.csv'

    # Configurazione dell'hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"eseguo training su {device}")
    # Inizializzazione dei parametri del modello
    initializer = m1.initializer_pytorch()
    # Inizializzazione del modello
    model = m1.build_model(trainingval=True, rateval=rateval)
    print(model)
    model = model.to(device)
    
    # Loss function (Binary Cross-Entropy Loss) ed ottimizzatore (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if train_test_from_scratch:
        # Prendi dati di input, di tipo numpy.ndarray()
        x_train, y_train = get_samples_labels(training_set_path)
        x_test, y_test = get_samples_labels(test_set_path)
        # Converti i dati in tensori
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        # Creazione dei DataLoader per il training e il test
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Calcola pos_weight
        num_pos = y_train.sum()  # Numero di esempi positivi (classe 1)
        num_neg = len(y_train) - num_pos  # Numero di esempi negativi (classe 0)
        pos_weight = torch.tensor(num_neg / num_pos).to(device)  # Calcolo del pos_weight
        # Loss function (Binary Cross-Entropy Loss) ed ottimizzatore (Adam)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Addestramento del modello
        trained_model = train_model(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            num_epochs=num_epochs, 
            save_path=local_path + 'trained_models/cnn_kepler_q1-q17_dr24_split100.pt'
            )
        # Valutazione del modello
        evaluate_model(trained_model, test_loader, criterion)
    else:
        print(f"Model assessment on {test_set_path}\n")
        # Carica modello addestrato
        model_path = local_path + 'trained_models/cnn_kepler_q1-q17_dr24_split90.pt'
        model.load_state_dict(torch.load(model_path))
        print(f"Carico modello {model_path}\n")
        # Loss senza class weighting
        criterion = nn.BCEWithLogitsLoss()
        # Prendi dati di input, di tipo numpy.ndarray()
        x_test, y_test = get_samples_labels(test_set_path)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Valuta il modello
        evaluate_model(model, test_loader, criterion)
