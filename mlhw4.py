import pandas as pd
import numpy as np
import re
import torch
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

#Task 1
df = pd.read_csv('movie_data.csv')
print(df.head())
print(df.shape)

nltk.download('stopwords')
stop = stopwords.words('english')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text += ' ' + ' '.join(emoticons).replace('-', '')
    return text

porter = PorterStemmer()
df['review'] = df['review'].apply(preprocessor)

X = df['review'].values
le = LabelEncoder()
y = le.fit_transform(df['sentiment'])  # y = 0 or 1

# Add tokenizer + stemming
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split() if word not in stop]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=True,
                        tokenizer=tokenizer_porter,
                        max_features=20000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"Train shape: {X_train_tfidf.shape}")
print(f"Test shape: {X_test_tfidf.shape}")

#task 2

#tensors
X_train_tensor = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

torch.manual_seed(1) #for reproducibility
# torch.set_num_threads(1)

##############################################
# 1. TEXTBOOK LOGISTIC REGRESSION
##############################################
print("Training textbook logistic regression...")
start_time = time.time()

# Textbook uses default params with increased max_iter
lr = LogisticRegression(penalty='l2',        
                       C=1.0,                
                       solver='liblinear',    
                       max_iter=1000,        
                       random_state=0)  

lr.fit(X_train_tfidf, y_train)
lr_pred = lr.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, lr_pred)
lr_time = time.time() - start_time

print(f"\n[Textbook] Logistic Regression Results:")
print(f"Accuracy: {lr_acc:.4f} (Expected: ~0.90)")
print(f"Training time: {lr_time:.2f}s")

##############################################
# 2. ENHANCED FNN WITH HYPERPARAMETER TUNING
##############################################
class FNN(nn.Module):
    def __init__(self, input_size, hidden_units=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 64)
        self.fc3 = nn.Linear(64, 2)
        # self.dropout = nn.Dropout(0.5) 
        
    def forward(self, x):
        x = F.relu(self.fc1(x))       
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        return self.fc3(x)

param_grid = {
    'lr': [0.001, 0.0005],           # Learning rates 
    'hidden_units': [128, 256],       # Hidden layer units 
    'weight_decay': [0.01, 0.001]     # L2 regularization 
}

best_acc = 0
best_model = None
best_params = None
print("\nTraining FNN with hyperparameter tuning...")

for lr in param_grid['lr']:
    for units in param_grid['hidden_units']:
        for wd in param_grid['weight_decay']:

            model = FNN(X_train_tensor.shape[1], hidden_units=units)
            optimizer = torch.optim.Adam(model.parameters(), 
                                       lr=lr, 
                                       weight_decay=wd)
            loss_fn = nn.CrossEntropyLoss()
            
            start_time = time.time()
            for epoch in range(10):  
                model.train()
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()
            end_time = time.time()
                
            model.eval()
            with torch.no_grad():
                    val_logits = model(X_test_tensor)
                    val_acc = (torch.argmax(val_logits, dim=1) == y_test_tensor).float().mean()
                    
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_model = model
                        best_params = f"lr={lr}, units={units}, wd={wd}"
                        best_time = end_time - start_time
            
            print(f"Config: lr={lr}, units={units}, wd={wd} | Val Acc: {val_acc:.4f} | Time: {end_time - start_time:.2f}s")

##############################################
# RESULTS COMPARISON
##############################################
print(f"1. Textbook Logistic Regression: {lr_acc:.4f} ({lr_time:.2f}s)")
print(f"2. [Without Dropout] Best FNN: {best_acc:.4f} (Params: {best_params}, Time: {best_time:.2f}s)")

