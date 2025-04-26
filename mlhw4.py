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
                        lowercase=True,
                        tokenizer=tokenizer_porter,
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
test_ds = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
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

train_accuracies = []
test_accuracies = []

for epoch in range(20):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

    #compute training accuracy
    model.eval()
    with torch.no_grad():
        train_preds = torch.argmax(model(X_train_tensor), dim=1)
        train_acc = accuracy_score(y_train, train_preds)
        train_accuracies.append(train_acc)

        test_logits = model(X_test_tensor)
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = accuracy_score(y_test, test_preds)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {time.time() - start_pt:.2f}s")

print(f"\n[PyTorch] Simple FNN Results:")
print(f"Final Train Accuracy: {train_accuracies[-1]:.4f}")
print(f"Final Test Accuracy: {test_accuracies[-1]:.4f}")
print(f"Training time: {time.time() - start_pt:.2f}s")

#plot train vs test accuracy
epochs = range(1, 21)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label='Simple FNN Train Accuracy')
plt.plot(epochs, test_accuracies, label='Simple FNN Test Accuracy')
plt.axhline(y=lr_acc, color='r', linestyle='--', label='Logistic Regression Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison: Simple FNN vs Logistic Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")



#################################################################################
# Enhanced FNN
#################################################################################

class FNN(nn.Module):
    def __init__(self, input_size, hidden_units=16, use_dropout=False, dropout_prob=0.3):
        super().__init__()
        self.use_dropout = use_dropout
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_units, 2)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        return self.fc2(x)


# # Hyperparameter tuning for FNN model
print("\nTuning FNN Model with hidden layer and dropout function...")
param_grid = {
    'lr': [0.01, 0.001, 0.0001],
    'weight_decay': [0.0, 0.0001, 0.00001]
}

def train_fnn_model(use_dropout):
    print(f"\nTraining FNN Model (Dropout = {use_dropout})...")
    start_time = time.time()

    best_acc = 0
    best_params = {}
    best_curve = []

    for lr_val in param_grid['lr']:
        for wd in param_grid['weight_decay']:
            model = FNN(X_train_tensor.shape[1], use_dropout=use_dropout)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=wd)
            loss_fn = nn.CrossEntropyLoss()
            epoch_acc = []

            for epoch in range(20):
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
                best_params = {'lr': lr_val, 'weight_decay': wd}
                best_curve = epoch_acc

            print(f"lr={lr_val}, weight_decay={wd}, Final Acc: {epoch_acc[-1]:.4f}")

    total_time = time.time() - start_time
    print(f"Best Accuracy: {best_acc:.4f} with params {best_params}, Time: {total_time:.2f}s")
    return best_curve, best_acc, best_params

nodrop_curve, nodrop_acc, nodrop_params = train_fnn_model(use_dropout=True)
dropout_curve, dropout_acc, dropout_params = train_fnn_model(use_dropout=False)


plt.figure(figsize=(12, 6))

# Simple FNN (linear) accuracies
epochs_simple = range(1, 21)
# plt.plot(epochs_simple, train_accuracies, label='Simple FNN Train Accuracy', linestyle='--')
plt.plot(epochs_simple, test_accuracies, label='Simple FNN Test Accuracy', linestyle='-')

# Tuned FNN (with hidden layer) test accuracy
epochs_hidden = range(1, 21)
plt.plot(epochs_hidden, nodrop_curve, label='Enhanced FNN (Hidden Layer) Test Accuracy', linestyle='-')

# Logistic Regression baseline (test accuracy)
plt.axhline(y=lr_acc, color='r', linestyle=':', label='Logistic Regression Test Accuracy')

# Plot setup
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs: Enhanced FNN with dropout vs Logistic Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_comparison_simp_enh.png")


############################################################################
# Task 3
############################################################################
def train_kfold(X_tensor, y_tensor, k=20, batch_size=32, epochs=20, lr=0.0001, weight_decay=0.00001):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accs = []
    times = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
        print(f"\nFold {fold+1}/{k}")
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = FNN(X_tensor.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        start = time.time()
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = torch.argmax(val_logits, dim=1)
            acc = accuracy_score(y_val, val_preds)
            accs.append(acc)
            times.append(time.time() - start)
            print(f"Fold {fold+1} Accuracy: {acc:.4f}, Time: {times[-1]:.2f}s")

    print(f"\nK-Fold Avg Accuracy: {np.mean(accs):.4f}, Avg Time per Fold: {np.mean(times):.2f}s")
    return accs, times

# With K-Fold:
accs_kfold, times_kfold = train_kfold(X_train_tensor, y_train_tensor, k=20)

print("\n--- Performance Comparison ---")
print(f"With K-Fold:    Accuracy = {np.mean(accs_kfold):.4f}, Time = {np.sum(times_kfold):.2f}s")


# Plotting the accuracies
plt.figure(figsize=(12, 6))

# Simple FNN (linear) accuracies
epochs_simple = range(1, 21)
plt.plot(epochs_simple, test_accuracies, label='Simple FNN Test Accuracy', linestyle='-')

# Tuned FNN (with hidden layer) test accuracy
# epochs_hidden = range(1, 21)
# plt.plot(epochs_hidden, test_accuracies, label='Enhanced FNN (Hidden Layer) Test Accuracy [with dropout]', linestyle='--')

# K-Fold accuracies 
plt.plot(range(1, len(accs_kfold) + 1), accs_kfold, label='K-Fold Test Accuracy', linestyle='-.')

# # Logistic Regression baseline (test accuracy)
# plt.axhline(y=lr_acc, color='r', linestyle=':', label='Logistic Regression Test Accuracy')

# Plot setup
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs: Simple FNN vs K-Fold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_comparison_kfold_all.png")

##########################################
# Task 5.2
##########################################

def train_single_dropout_model(dropout_prob, lr=0.0001, weight_decay=0.0001, seed=1):
    torch.manual_seed(seed)
    model = FNN(X_train_tensor.shape[1], dropout_prob=dropout_prob)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    return model

def evaluate_bagging_ensemble(models, X_test_tensor, y_test):
    all_logits = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(X_test_tensor)
            all_logits.append(logits)

    # Average logits
    avg_logits = torch.stack(all_logits).mean(dim=0)
    preds = torch.argmax(avg_logits, dim=1)
    acc = accuracy_score(y_test, preds)
    return acc

dropout_probs = [0.2, 0.3, 0.4, 0.5, 0.6]
seeds = [1, 2, 3, 4, 5] #for reproducibility
bagged_models = []

for prob, seed in zip(dropout_probs, seeds):
    print(f"Training dropout model with prob={prob}, seed={seed}")
    model = train_single_dropout_model(dropout_prob=prob, seed=seed)
    bagged_models.append(model)

# Evaluate each model separately
individual_accuracies = []
for i, model in enumerate(bagged_models):
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y_test, preds)
        individual_accuracies.append(acc)
        print(f"Model {i+1} Accuracy: {acc:.4f} | Model {i+1} Time: {time.time() - start_time:.2f}s")

ensemble_acc = evaluate_bagging_ensemble(bagged_models, X_test_tensor, y_test)
print(f"\nBagging Ensemble Accuracy (5 dropout models): {ensemble_acc:.4f}")
print(f"Baseline No-Dropout Accuracy: {dropout_acc:.4f}")
print(f"Single Dropout Model Accuracy: {nodrop_acc:.4f}")


# Plot
plt.figure(figsize=(8, 5))
plt.bar([f"Model {i+1}" for i in range(len(individual_accuracies))], individual_accuracies, label="Individual Models", color="skyblue")
plt.axhline(y=ensemble_acc, color='r', linestyle='--', label=f"Ensemble ({ensemble_acc:.4f})")
plt.title("Test Accuracies of Dropout Models vs. Ensemble")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig("ensemble_accuracy_comparison.png")
