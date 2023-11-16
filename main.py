# used a dictionary to represent an intents JSON file
data = {"intents": [
    {"tag": "greeting",
     "responses": ["Howdy Partner!", "Hello", "How are you doing?", "Greetings!", "How do you do?"]},
    {"tag": "age",
     "responses": ["I am 25 years old", "I was born in 1998", "My birthday is July 3rd and I was born in 1998", "03/07/1998"]},
    {"tag": "date",
     "responses": ["I am available all week", "I don't have any plans", "I am not busy"]},
    {"tag": "name",
     "responses": ["My name is James", "I'm James", "Hey!! there I am James"]},
    {"tag": "goodbye",
     "responses": ["It was nice speaking to you", "See you later", "Speak soon!"]},
    {"tag": "thanks",
     "responses": ["You're welcome!", "No problem!", "Glad I could help!"]},
    {"tag": "help",
     "responses": ["How can I assist you?", "What do you need help with?", "I'm here to help!"]},
    {"tag": "jokes",
     "responses": ["Why did the chicken cross the road? To get to the other side!", "Knock, knock. Who's there?", "Here's a joke: What's orange and sounds like a parrot? A carrot!"]},
    {"tag": "music",
     "responses": ["I love music! What's your favorite genre?", "Music is great for the soul.", "I can't sing, but I can tell you about music!"]},
    {"tag": "weather",
     "responses": ["The weather is nice today.", "I can look up the weather for you. Where are you located?", "Sunny days are the best!"]},
    {"tag": "movies",
     "responses": ["What's your favorite movie genre?", "I enjoy watching movies too.", "Movies are a great way to relax."]}
]}
import gradio as gr
import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt

# We have prepared a chitchat dataset with 5 labels
df = pd.read_csv(r'datasetforchatbot.csv')
df.head()

df['label'].value_counts()

# Converting the labels into encodings
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
# check class distribution
df['label'].value_counts(normalize = True)

# In this example we have used all the utterances for training purpose
train_text, train_labels = df['text'], df['label']

from transformers import AutoModel, BertTokenizerFast
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# Import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

from transformers import DistilBertTokenizer, DistilBertModel
# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Import the DistilBert pretrained model
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

text = ["this is a distil bert model.","data is oil"]
# Encode the text
encoded_input = tokenizer(text, padding=True,truncation=True, return_tensors='pt')
print(encoded_input)
# In input_ids:
# 101 - Indicates beginning of the sentence
# 102 - Indicates end of the sentence
# In attention_mask:
# 1 - Actual token
# 0 - Padded token

# get length of all the messages in the train set
seq_len = [len(i.split()) for i in train_text]
pd.Series(seq_len).hist(bins = 10)
# Based on the histogram we are selecting the max len as 8
max_seq_len = 8

# tokenize and encode sequences in the training set
tokens_train = tokenizer(
    train_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#define a batch size
batch_size = 16
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# DataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

class BERT_Arch(nn.Module):
   def __init__(self, bert):
       super(BERT_Arch, self).__init__()
       self.bert = bert

       # dropout layer
       self.dropout = nn.Dropout(0.2)

       # relu activation function
       self.relu =  nn.ReLU()
       # dense layer
       self.fc1 = nn.Linear(768,512)
       self.fc2 = nn.Linear(512,256)
       self.fc3 = nn.Linear(256,5)
       #softmax activation function
       self.softmax = nn.LogSoftmax(dim=1)
       #define the forward pass
   def forward(self, sent_id, mask):
      #pass the inputs to the model
      cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]

      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)

      x = self.fc2(x)
      x = self.relu(x)
      x = self.dropout(x)
      # output layer
      x = self.fc3(x)

      # apply softmax activation
      x = self.softmax(x)
      return x

# freeze all the parameters. This will prevent updating of model weights during fine-tuning.
for param in bert.parameters():
      param.requires_grad = False
model = BERT_Arch(bert)
from torchinfo import summary
summary(model)

from transformers import AdamW
# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)

from sklearn.utils.class_weight import compute_class_weight
#compute the class weights
class_wts = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
print(class_wts)

# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
# loss function
cross_entropy = nn.NLLLoss(weight=weights)

import torch.optim.lr_scheduler as lr_scheduler
# empty lists to store training and validation loss of each epoch
train_losses=[]
# number of training epochs
epochs = 50
# We can also use learning rate scheduler to achieve better results
lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

import numpy as np
import torch
from torch.nn.functional import cross_entropy

# Function to compute class weights
def compute_class_weights(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    class_weights = total_samples / (len(unique_labels) * counts)
    return class_weights

def train():
    model.train()
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        sent_id, mask, labels = batch
        preds = model(sent_id, mask)

        # Compute the unweighted loss
        loss = cross_entropy(preds, labels)

        # Compute class weights based on training labels
        class_weights = torch.tensor(compute_class_weights(labels), dtype=torch.float32)

        # Apply class weights to the loss
        weighted_loss = (loss * class_weights).mean()

        total_loss = total_loss + weighted_loss.item()

        # Backpropagation and optimization
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_preds.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / (step + 1)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

# Set your random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    # Train the model and get the average loss and predictions
    train_loss, _ = train()

    # Convert train_loss to a NumPy array
    train_loss = train_loss  # No need for .item()

    # Append training loss
    train_losses.append(train_loss)

print(f'\nTraining Loss: {train_loss:.3f}')

def get_prediction(str):
 str = re.sub(r'[^a-zA-Z ]+', '', str)
 test_text = [str]
 model.eval()

 tokens_test_data = tokenizer(
 test_text,
 max_length = max_seq_len,
 pad_to_max_length=True,
 truncation=True,
 return_token_type_ids=False
 )
 test_seq = torch.tensor(tokens_test_data['input_ids'])
 test_mask = torch.tensor(tokens_test_data['attention_mask'])

 preds = None
 with torch.no_grad():
   preds = model(test_seq, test_mask)
 preds = preds.detach().cpu().numpy()
 preds = np.argmax(preds, axis = 1)
 print("Intent Identified: ", le.inverse_transform(preds)[0])
 return le.inverse_transform(preds)[0]
def get_response(message):
  intent = get_prediction(message)
  for i in data['intents']:
    if i["tag"] == intent:
      result = random.choice(i["responses"])
      break
  print(f"Response : {result}")
  return "Intent: "+ intent + '\n' + "Response: " + result

get_response('Hi')