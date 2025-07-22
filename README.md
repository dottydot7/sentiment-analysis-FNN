The dataset consists of movie reviews paired with their corresponding sentiment labels. After preprocessing - which involves removing HTML tags, handling emoticons, converting text to lowercase, stripping punctuation, and eliminating stop words - the cleaned reviews are transformed into a TF-IDF representation. The resulting matrices have the following dimensions: 

- Training set: 35,000 samples × 20,000 features 
- Test set: 15,000 samples × 20,000 features 

We split the data into a standard 70-30 ratio (training vs. testing) using train_test_split from scikit-learn. The TF-IDF features and labels are then converted into PyTorch tensors to prepare them for learning. For efficient training, we use PyTorch’s DataLoader to create mini-batches, allowing the model to process smaller chunks of data at a time rather than the entire dataset at once. 




 
  


