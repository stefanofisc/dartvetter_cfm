import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchsummary import summary
#from ForestDiffusion import ForestDiffusionModel as ForestFlowModel

def initializer_pytorch():
    # Utilizza Xavier Uniform per inizializzare i pesi
    return nn.init.xavier_uniform_

class GlobalViewFiscale(nn.Module):
    def __init__(self, initializer, rateval):
        super(GlobalViewFiscale, self).__init__()
        
        # Global view input
        self.GV_LENGTH = 201
        
        # Blocks configuration
        self.conv_blocks_num = 5
        self.conv_filter_size = 3
        self.conv_block_filter_factor = 2
        self.pooling_size = 5
        self.pooling_stride = 2
        
        # Layers
        self.convs = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(self.conv_blocks_num):
            in_channels = 1 if i == 0 else pow(self.conv_block_filter_factor, 4 + (i - 1))
            out_channels = pow(self.conv_block_filter_factor, 4 + i)
            
            self.convs.append(nn.Conv1d(in_channels, out_channels, self.conv_filter_size, padding='same'))
            self.dropout.append(nn.Dropout(rateval))
            self.batch_norms.append(nn.BatchNorm1d(out_channels))

        self.flatten = nn.Flatten()
        
    def forward(self, x, trainingval):
        x = x.view(-1, 1, self.GV_LENGTH)  # Reshape input to (batch_size, GV_LENGTH, 1)
        
        for i in range(self.conv_blocks_num):
            x = self.convs[i](x)
            x = F.relu(x)
            x = self.dropout[i](x) if trainingval else x  # Apply dropout conditionally
            x = F.max_pool1d(x, kernel_size=self.pooling_size, stride=self.pooling_stride)
            x = self.batch_norms[i](x)
        
        output = self.flatten(x)
        return output


class FullyConnectedFiscale(nn.Module):
    def __init__(self, initializer, input_size, rateval):
        super(FullyConnectedFiscale, self).__init__()
        
        self.fc_units = 512 # dimensione dell'output prodotto dal blocco di estrazione di features
        self.fc_layer = nn.Linear(input_size, self.fc_units)
        self.dropout = nn.Dropout(rateval)
        self.output_layer = nn.Linear(self.fc_units, 1)
        

    def forward(self, x):
        x = F.relu(self.fc_layer(x))
        x = self.dropout(x)
        #x = torch.sigmoid(self.output_layer(x)) # Rimuovo sigmoid per usare BCEWithLogitsLoss per migliore stabilità numerica
        return self.output_layer(x)


class DartVetterFiscale2024(nn.Module):
    def __init__(self, initializer, rateval, trainingval):
        super(DartVetterFiscale2024, self).__init__()
        self.global_view = GlobalViewFiscale(initializer, rateval)
        self.global_view_output_size = 768
        self.fully_connected = FullyConnectedFiscale(initializer, self.global_view_output_size, rateval)


    def forward(self, x, trainingval=True):
        output_1 = self.global_view(x, trainingval)
        output = self.fully_connected(output_1)

        return output


def build_model(trainingval=True, rateval=0.3):    
    # Initialization of model's parameters
    initializer = initializer_pytorch()
    
    model = DartVetterFiscale2024(initializer, rateval, trainingval)
    
    return model


class ModelInspector:
    """
        # Esempio di utilizzo
    >>> if __name__ == '__main__':
    >>>    # Supponendo che `model` sia un'istanza del tuo modello
    >>>    model = DartVetterFiscale2024(initializer, rateval, trainingval)
        
    >>>    inspector = ModelInspector(model)
        
    >>>    # Conteggio dei parametri addestrabili
    >>>    num_params = inspector.count_trainable_params()
    >>>    print(f"Numero totale di parametri addestrabili: {num_params}")

    >>>    # Stampa i parametri di ogni layer
    >>>    inspector.print_layer_params()

    >>>    # Stampa un riepilogo dell'architettura del modello
    >>>    inspector.print_model_summary()
    """
    def __init__(self, model):
        self.model = model

    def count_trainable_params(self):
        """Contare il numero totale di parametri addestrabili nel modello."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_layer_params(self):
        """Stampare il numero di parametri di ogni livello nel modello."""
        print(f"{'Layer':<30} {'Param Count':<20}")
        print("="*50)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name:<30} {param.numel():<20}")

    def print_model_summary(self):
        """Stampare l'architettura e i dettagli del modello."""
        # Assicurati che il modello sia caricato su un dispositivo
        device = next(self.model.parameters()).device
        summary(self.model.to(device), input_size=(1, 201))



class MetricTracker:
    def __init__(self):
        self.precision_metric = torchmetrics.Precision(task='binary', num_classes=1, average='macro')
        self.recall_metric = torchmetrics.Recall(task='binary', num_classes=1, average='macro')
        self.f1_metric = torchmetrics.F1Score(task='binary', num_classes=1, average='macro')
        self.losses = []
        self.accuracies = []

    def update(self, outputs, labels, loss):
        # Calcola le metriche e aggiornale
        self.losses.append(loss.item())
        preds = torch.round(torch.sigmoid(outputs).squeeze(-1)) # Rimuovi la dimensione extra cosìcchè preds abbia la stessa forma dei veri labels
        accuracy = (preds == labels.unsqueeze(1)).float().mean()
        self.accuracies.append(accuracy.item())

        # Aggiorna le metriche di precisione, richiamo e f1
        self.precision_metric(preds, labels)
        self.recall_metric(preds, labels)
        self.f1_metric(preds, labels)

    def compute(self):
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        f1 = self.f1_metric.compute()

        # Reset delle metriche per il prossimo calcolo
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()

        return precision.item(), recall.item(), f1.item()


if __name__ == '__main__':
    trainingval = True
    my_model = build_model()
    print(my_model)