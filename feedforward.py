from binaryclassifier import gradient_descent, predict
import numpy as np
import pandas as pd
import torch
from torch import tensor
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

    
def cabin_deck(row):
    return row['Cabin'].split('/')[0] if row['Cabin'] != '???' else '???'


def cabin_side(row):
    return row['Cabin'].split('/')[2] if row['Cabin'] != '???' else '???'


def surname(row):
    return row['Name'].split(' ')[1] if row['Name'] != '???' else '???'


def surname_initial(row):
    return (row['Name'].split(' ')[1] + "****")[0] if row['Name'] != '???' else '???'


# feature from anonymous student
def compute_common_surnames():
    df = pd.read_csv('data/train.csv')
    names = dict()
    for _, row in df.iterrows():
        surname = row['Name'].split()[1]
        if surname not in names:
            names[surname] = 0
        names[surname] += 1
    common_names = set()  
    for name in names:
        if names[name] > 3:
            common_names.add(name)
    return common_names


COMMON_NAMES = compute_common_surnames()


def common_surname(row):
    return surname(row) in COMMON_NAMES


def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    data['Deck'] = data.apply(cabin_deck, axis=1)  # axis=1 means operate on rows
    data['Side'] = data.apply(cabin_side, axis=1)  
    data['Surname'] = data.apply(surname, axis=1)
    data['SurnameInitial']  = data.apply(surname_initial, axis=1) 
    data['CommonSurname'] = data.apply(common_surname, axis=1)
    return data
  
  
def categorical_feature(row, feature_name, domain):
    domain = domain + ['Other']
    result = [0.0] * len(domain)
    if row[feature_name] in domain:
        result[domain.index(row[feature_name])] = 1.0
    else:
        result[domain.index('Other')] = 1.0    
    return tensor(result)


def numeric_feature(row, feature_name, mean, stddev):
    feat = 0.0 if np.isnan(row[feature_name]) else (float(row[feature_name]) - mean) / stddev    
    return tensor([feat])


ALL_LETTERS = 'abcdefghijklmnopqrstuvwxyz*'


def surname_features(feature_dict, max_word_length=12):
    word_matrices = []
    word = feature_dict['Surname']
    word = word[:max_word_length]
    word = word + '*'*(max_word_length-len(word))         
    word_matrix = torch.zeros(max_word_length, len(ALL_LETTERS))
    word_matrices.append(word_matrix)
    for i, character in enumerate(word):
        word_matrix[i][ALL_LETTERS.index(character.lower())] = 1            
    return torch.stack(word_matrices).squeeze()

    
def categorical_feature(row, feature_name, domain):
    domain = domain + ['Other']
    result = [0.0] * len(domain)
    if row[feature_name] in domain:
        result[domain.index(row[feature_name])] = 1.0
    else:
        result[domain.index('Other')] = 1.0    
    return tensor(result)


def numeric_feature(row, feature_name, mean, stddev):
    feat = 0.0 if np.isnan(row[feature_name]) else (float(row[feature_name]) - mean) / stddev    
    return tensor([feat])


def tabular_features(row):
    PLANETS = ['Earth', 'Europa', 'Mars']
    BOOLEANS = [False, True]
    DESTINATIONS = ['55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e']
    CABIN_DECKS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', '???']
    CABIN_SIDES = ['P', 'S', '???']
    LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ*')
    return torch.cat([
        categorical_feature(row, 'HomePlanet', PLANETS),
        categorical_feature(row, 'CryoSleep', BOOLEANS),
        categorical_feature(row, 'Destination', DESTINATIONS),
        categorical_feature(row, 'VIP', BOOLEANS),  
        categorical_feature(row, 'Deck', CABIN_DECKS),          
        categorical_feature(row, 'Side', CABIN_SIDES),        
        categorical_feature(row, 'SurnameInitial', LETTERS), 
        categorical_feature(row, 'CommonSurname', BOOLEANS),
        numeric_feature(row, 'Age', 40.0, 40.0),
        numeric_feature(row, 'RoomService', 223.0, 662.0),
        numeric_feature(row, 'FoodCourt', 465.0, 1654.0),
        numeric_feature(row, 'ShoppingMall', 171.0, 553.0),
        numeric_feature(row, 'Spa', 313.0, 1146.0),  
        numeric_feature(row, 'VRDeck', 302.0, 1131.0)
    ], dim=0)
 

class SpaceshipZitanicData(Dataset):
    def __init__(self, csv_file, test_set=False):
        super().__init__()
        data = read_csv(csv_file)
        self.x = []
        self.y = []
        for _, row in tqdm(data.iterrows()):              
            self.x.append({f: row[f] for f in set(data.columns) - {'Transported'}})
            if test_set:
                self.y.append(-1)
            else:
                self.y.append(row['Transported'])   
        self.num_tabular_features = tabular_features(self.x[0]).shape[0]     
                      
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return (tabular_features(self.x[idx]), self.y[idx])  
    
    
class NeuralNetwork(torch.nn.Module):
    
    def __init__(self, H, num_features):
        super().__init__()
        self.H = H
        self.theta_dot = Parameter(torch.zeros(H, num_features))
        self.theta_ddot = Parameter(torch.zeros(1, H))
        for param in [self.theta_dot, self.theta_ddot]:
            torch.nn.init.kaiming_uniform_(param)
            
    def forward(self, x):
        result = self.theta_dot @ x.t()
        result = torch.relu(result)
        result = self.theta_ddot @ result
        result = result.squeeze()
        return torch.sigmoid(result) 
   

if __name__ == "__main__":
    print("Loading data...")
    train_set = SpaceshipZitanicData('data/train.csv')
    test_set = SpaceshipZitanicData('data/dev.csv')       
    num_epochs = 60
    model = NeuralNetwork(100, train_set.num_tabular_features)
    trained_model = gradient_descent(model, num_epochs, train_set, test_set)        
    
    test_set = SpaceshipZitanicData('data/test.csv', test_set=True)     
    test_loader = DataLoader(test_set, batch_size=512)          
    results = predict(model, test_loader)
    with open('predictions.txt', 'w') as writer:           
        for result in results:
            writer.write(str(result.item()) + '\n')