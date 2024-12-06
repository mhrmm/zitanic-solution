from binaryclassifier import gradient_descent, predict
import pandas as pd
import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm


def surname(row):
    return row['Name'].split(' ')[1] if row['Name'] != '???' else 'X'


def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    data['Surname'] = data.apply(surname, axis=1) 
    return data
  

def surname_features(feature_dict, max_word_length=12):
    ALL_LETTERS = 'abcdefghijklmnopqrstuvwxyz*'
    word_matrices = []
    word = feature_dict['Surname']
    word = word[:max_word_length]
    word = word + '*'*(max_word_length-len(word))         
    word_matrix = torch.zeros(max_word_length, len(ALL_LETTERS))
    word_matrices.append(word_matrix)
    for i, character in enumerate(word):
        word_matrix[i][ALL_LETTERS.index(character.lower())] = 1            
    return torch.stack(word_matrices).squeeze()


class SpaceshipZitanicData(torch.nn.Module):
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
                      
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return (surname_features(self.x[idx]), self.y[idx])  
    
 
class BatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=1).unsqueeze(dim=1)
        stddev = torch.std(x, dim=1).unsqueeze(dim=1)
        result = (x - mean) / stddev
        return result
    
            
class RNN(torch.nn.Module):
    def __init__(self, H, V, use_batch_norm=True):
        super().__init__()
        self.H = H
        self.theta = Parameter(torch.zeros(H, H+V))
        self.bias = Parameter(torch.zeros(1))
        self.theta_dot = Parameter(torch.zeros(1, H))
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = BatchNorm()
        for param in [self.theta, self.theta_dot]:
            torch.nn.init.kaiming_uniform_(param)
            
    def forward(self, sequential_features):
        state = torch.zeros(sequential_features.shape[0], self.H)
        for i in range(sequential_features.shape[1]):
            token = sequential_features[:,i,:]
            state_and_token = torch.cat([state, token], dim=1).unsqueeze(dim=2)    
            z = (self.theta @ state_and_token).squeeze() + self.bias
            if self.use_batch_norm:
                z = self.bn(z)
            state = torch.relu(z)        
        result = self.theta_dot @ state.unsqueeze(dim=2)
        result = result.squeeze()
        return torch.sigmoid(result) 
    

if __name__ == "__main__":
    print("Loading data...")
    train_set = SpaceshipZitanicData('data/train.csv')
    test_set = SpaceshipZitanicData('data/dev.csv')       
    num_epochs = 100
    model = RNN(H=100, V=27)  # 27 = the english alphabet + pad token
    trained_model = gradient_descent(model, num_epochs, train_set, test_set)        
        
    test_set = SpaceshipZitanicData('data/test.csv', test_set=True)     
    test_loader = DataLoader(test_set, batch_size=512)          
    results = predict(model, test_loader)
    with open('predictions.txt', 'w') as writer:           
        for result in results:
            writer.write(str(result.item()) + '\n')