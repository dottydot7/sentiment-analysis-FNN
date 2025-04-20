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

torch.set_num_threads(1)
#Task 1
df = pd.read_csv('C:\\Users\\Elekt\\spring25\\cs529\\mlhw4\\movie_data.csv')
print(df.head())
print(df.shape)

nltk.download('stopwords')
stop = stopwords.words('english')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

porter = PorterStemmer()
df['review'] = df['review'].apply(preprocessor)

X = df['review'].values
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None,
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

class FNN(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 100)
        self.fc2 = torch.nn.Linear(100, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x) 

# softmax function (from notes)
def softmax(z):
    return torch.exp(z) / torch.sum(torch.exp(z), dim=1, keepdim=True)

# Initialize
#model = FNN(X_train_tensor.shape[1])
learning_rates = [0.1, 0.01, 0.001]
weight_decays = [1e-4, 1e-5, 0]
#optimizer = torch.optim.Adam(model.parameters(), learning_rate)
#loss_fn = torch.nn.CrossEntropyLoss()
best_acc = 0
best_params = {}

for lr in learning_rates:
    for wd in weight_decays:
        print(f"\nTesting lr={lr}, weight_decay={wd}")
        model = FNN(X_train_tensor.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = torch.nn.CrossEntropyLoss()
        # Training with gradient update
        num_epochs = 5
        for epoch in range(num_epochs):
            for xb, yb in train_loader:
                # reset gradients
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                #update model weights
                optimizer.step()

                with torch.no_grad():
                    for param in model.parameters():
                        param -= lr * param.grad
                        param.grad.zero_()

            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")



        with torch.no_grad():
            logits = model(X_test_tensor)
            probs = softmax(logits)
            predicted_labels = torch.argmax(probs, dim=1)
            acc = (predicted_labels == y_test_tensor).float().mean()
            print(f"Test Accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_params = {'lr':lr, 'wd': wd}

print(f"\nBest accuracy: {best_acc:.4f}")
print(f"Best parameters: {best_params}")
