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
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

#Task 1
# Download NLTK stopwords
nltk.download('stopwords')

# Load dataset
print("Loading movie dataset...")
df = pd.read_csv('movie_data.csv')
print(df.head())
print(f"Dataset shape: {df.shape}")

# Preprocessing steps
stop = stopwords.words('english')
porter = PorterStemmer()

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub(r'[\W]+', ' ', text.lower())
    text += ' ' + ' '.join(emoticons).replace('-', '')
    return text

# Apply preprocessing
df['review'] = df['review'].apply(preprocessor)

X = df['review'].values
y = df['sentiment'].values

# Add tokenizer + stemming
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split() if word not in stop]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        tokenizer=None,
                        max_features=20000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"TF-IDF Shapes => Train: {X_train_tfidf.shape}| Test: {X_test_tfidf.shape}")

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
# TEXTBOOK LOGISTIC REGRESSION
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

################################################
# Task 2 - Simple FNN
################################################

class SimpleFNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 2)

    def forward(self, x):
        return self.fc(x)


# Train linear PyTorch model
print("\nTraining Linear Model (PyTorch)...")
model = SimpleFNN(X_train_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

pt_accuracies = []
start_pt = time.time()
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y_test, preds)
        pt_accuracies.append(acc)
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Accuracy: {acc:.4f}")

pt_total_time = time.time() - start_pt
print(f"\nFinal PyTorch Accuracy: {pt_accuracies[-1]:.4f}, Time: {pt_total_time:.2f}s")

#################################################################################
# Enhanced FNN
#################################################################################

class FNN(nn.Module):
    def __init__(self, input_size, hidden_units=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 2)
        # self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        return self.fc2(x)


# Hyperparameter tuning for FNN model
print("\nTuning FNN Model...")
param_grid = {
    'lr': [0.01, 0.001, 0.0001],
    'weight_decay': [0.0, 0.001, 0.0001]
}

best_acc = 0
best_model = None
best_params = {}
accuracies_by_epoch = []
start_tune = time.time()

for lr_val in param_grid['lr']:
    for wd in param_grid['weight_decay']:
        model = FNN(X_train_tensor.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=wd)
        loss_fn = nn.CrossEntropyLoss()
        epoch_acc = []

        for epoch in range(30):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(X_test_tensor)
                preds = torch.argmax(logits, dim=1)
                acc = accuracy_score(y_test, preds)
                epoch_acc.append(acc)

        if epoch_acc[-1] > best_acc:
            best_acc = epoch_acc[-1]
            best_model = model
            best_params = {'lr': lr_val, 'weight_decay': wd}
            accuracies_by_epoch = epoch_acc

        print(f"lr={lr_val}, weight_decay={wd}, Accuracy: {epoch_acc[-1]:.4f}")

tune_time = time.time() - start_tune
print(f"\nBest Accuracy: {best_acc:.4f} with params {best_params}, Time: {tune_time:.2f}s")

# Plot accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, 31), accuracies_by_epoch, marker='o', label='PyTorch FNN')
plt.axhline(y=lr_acc, color='r', linestyle='--', label='Logistic Regression (sklearn)')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Accuracy over Epochs (Best FNN Config)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#############################################################################
# Task 3
#############################################################################

# Convert data to numpy for KFold splitting
X_np = X_train_tfidf.toarray()  
y_np = y_train

# Initialize KFold
k = 10  
kf = KFold(n_splits=k, shuffle=True, random_state=0)
print(f"\nTraining FNN using {k}-Fold Cross Validation...[Without Dropout]")

# Store results
fold_accuracies = []
fold_times = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
    print(f"\nFold {fold + 1}/{k}")
    
    # Split data
    X_train_fold, X_val_fold = X_np[train_idx], X_np[val_idx]
    y_train_fold, y_val_fold = y_np[train_idx], y_np[val_idx]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long)
    
    # DataLoader
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # Model and optimizer
    model = FNN(X_train_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    
    # Training
    start_time = time.time()
    for epoch in range(10):  # 10 epochs per fold
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.cross_entropy(pred, yb)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_acc = (torch.argmax(val_logits, dim=1) == y_val_tensor).float().mean()
    
    fold_accuracies.append(val_acc.item())
    fold_time = time.time() - start_time
    fold_times.append(fold_time)
    print(f"Fold Acc: {val_acc:.4f}, Time: {fold_time:.2f}s")

# Results
print(f"\nMean CV Accuracy: {np.mean(fold_accuracies):.4f}" | f"Mean Time per Fold: {np.mean(fold_times):.2f}s" )
