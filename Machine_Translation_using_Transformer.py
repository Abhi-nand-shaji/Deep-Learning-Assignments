


import pandas as pd
import re

# Load dataset
df = pd.read_csv("team18_ta_train.csv")

# Drop empty rows
df.dropna(subset=['source', 'target'], inplace=True)

# Function to clean text
def clean_text(text, language="en"):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    
    if language == "en":
        # Remove non-alphanumeric (keep basic punctuation)
        text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\-\s]", '', text)
        text = text.lower()
    elif language == "ta":
        # Remove non-Tamil characters (keep punctuation optionally)
        # Unicode range for Tamil: 0B80–0BFF
        text = re.sub(r'[^\u0B80-\u0BFF\s.,!?]', '', text)
    
    return text

# Apply cleaning
df['source'] = df['source'].apply(lambda x: clean_text(x, language='en'))
df['target'] = df['target'].apply(lambda x: clean_text(x, language='ta'))

# Optional: Remove misaligned pairs (e.g., length ratio > 3)
def is_reasonable_pair(src, tgt, ratio=3.0):
    return 1/ratio < (len(src.split()) / (len(tgt.split()) + 1e-5)) < ratio

df = df[df.apply(lambda row: is_reasonable_pair(row['source'], row['target']), axis=1)]

# Save cleaned version
df.to_csv("cleaned_train.csv", index=False)


# ### Tokenization and Vocabulary Building



import pandas as pd

df = pd.read_csv("cleaned_train.csv")
df2=pd.read_csv("team18_ta_valid.csv")
english_sentences = df['source'].tolist()
tamil_sentences = df['target'].tolist()




# Simple whitespace tokenization (or use nltk.word_tokenize)
english_tokenized = [sentence.lower().split() for sentence in english_sentences]




### figuring out the max length of tokenization
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# Load IndicBERT tokenizer
tam_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

# Load Tamil sentences
df = pd.read_csv("cleaned_train.csv")
tamil_sentences = df['target'].tolist()

# Get token lengths for each sentence
token_lengths = [len(tam_tokenizer.tokenize(sentence)) for sentence in tamil_sentences]

# Compute statistics
max_len = max(token_lengths)
mean_len = np.mean(token_lengths)
median_len = np.median(token_lengths)
p95_len = np.percentile(token_lengths, 95)

print(f"Max length      : {max_len}")
print(f"Mean length     : {mean_len:.2f}")
print(f"Median length   : {median_len}")
print(f"95th percentile : {p95_len:.0f}")





from transformers import AutoTokenizer

tam_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
tamil_tokenized = [tam_tokenizer(sentence, truncation=True, padding='max_length', max_length=64, return_tensors="pt")["input_ids"].squeeze(0) for sentence in tamil_sentences]



from collections import Counter

def build_vocab(tokenized_sentences, min_freq=1):
    """
    Builds a vocab dict mapping word -> index
    
    Args:
        tokenized_sentences: List of List of tokens (words)
        min_freq: Minimum frequency to keep a word in vocab
        
    Returns:
        vocab: dict {word: idx}
        word_freq: dict {word: frequency}
    """
    counter = Counter()
    for sentence in tokenized_sentences:
        counter.update(sentence)
    
    # Special tokens
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }
    
    idx = len(vocab)
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1
            
    return vocab, counter





def load_glove_embeddings1(glove_file_path):
    glove_dict = {}
    with open(glove_file_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = [float(x) for x in parts[1:]]
            glove_dict[word] = vector
    return glove_dict

glove_dict = load_glove_embeddings1("glove.6B.200d.txt")

# Add an <unk> token with random values (optional)
import numpy as np
glove_dict["<unk>"] = list(np.random.uniform(-0.1, 0.1, 200))




import numpy as np

def load_glove_embeddings(glove_path, vocab, embedding_dim=200):
    """
    Load GloVe vectors for words in vocab, create embedding matrix
    
    Args:
        glove_path: Path to glove.6B.200d.txt
        vocab: word to index dict
        embedding_dim: dimension of GloVe vectors
        
    Returns:
        embedding_matrix: np.array of shape (vocab_size, embedding_dim)
    """
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings_index[word] = vector
    
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    # Initialize <unk> and others with random or zeros
    embedding_matrix[vocab["<unk>"]] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    for word, idx in vocab.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        elif word not in ["<pad>", "<unk>", "<bos>", "<eos>"]:
            # Random init for words not in GloVe
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    return embedding_matrix



# Suppose english_tokenized is your list of tokenized English sentences
import torch

# Build vocab
vocab, word_freq = build_vocab(english_tokenized, min_freq=1)
idx2word = {idx: word for word, idx in vocab.items()}
print(f"Vocabulary size: {len(vocab)}")

# Load GloVe embeddings and build embedding matrix
glove_path = "glove.6B.200d.txt"
embedding_matrix = load_glove_embeddings(glove_path, vocab, embedding_dim=200)
embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
print(f"Embedding matrix shape: {embedding_matrix.shape}")





tamil_vocab = tam_tokenizer.get_vocab()  # dict: token → id
id_to_token = {v: k for k, v in tamil_vocab.items()}



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np




def glove_tokenizer(sentence):
    return sentence.lower().strip().split()





indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
indic_model = AutoModel.from_pretrained("ai4bharat/indic-bert")
indic_model.eval()  # freeze unless you plan to fine-tune

for param in indic_model.parameters():
    param.requires_grad = False





class TranslationDataset(Dataset):
    def __init__(self, df, eng_vocab, glove_tokenizer, max_len=64):
        self.df = df.reset_index(drop=True)
        self.eng_vocab = eng_vocab
        self.eng_pad_id = eng_vocab["<pad>"]
        self.tokenizer = indic_tokenizer
        self.max_len = max_len
        self.glove_tokenizer = glove_tokenizer

    def encode_english(self, sentence):
        tokens = self.glove_tokenizer(sentence.lower())
        ids = [self.eng_vocab.get(tok, self.eng_vocab["<unk>"]) for tok in tokens]
        ids = ids[:self.max_len]
        padding = [self.eng_pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids + padding)

    def encode_tamil(self, sentence):
        out = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = self.df.loc[idx, 'source']
        tgt = self.df.loc[idx, 'target']
        src_ids = self.encode_english(src)
        tgt_ids, tgt_mask = self.encode_tamil(tgt)
        return src_ids, tgt_ids, tgt_mask





eng_vocab=vocab
train_dataset = TranslationDataset(
    df=df,
    eng_vocab=eng_vocab,
    glove_tokenizer=glove_tokenizer,
    max_len=64  # or whatever fits your model
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)


valid_dataset = TranslationDataset(
    df=df2,
    eng_vocab=eng_vocab,
    glove_tokenizer=glove_tokenizer,
    max_len=64  # or whatever fits your model
)

val_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)





class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_embedding_matrix,
                 src_embed_dim=200, tgt_embed_dim=768, nhead=4, num_encoder_layers=3, num_decoder_layers=3, ff_dim=512, dropout=0.1):
        super().__init__()
        

        # GloVe for English
        self.src_embedding = nn.Embedding.from_pretrained(src_embedding_matrix, freeze=True)
        self.pos_encoder = PositionalEncoding(src_embed_dim)

        # IndicBERT embeddings will be injected directly (not learned here)
        self.pos_decoder = PositionalEncoding(tgt_embed_dim)

        # Encoder: uses src_embed_dim
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=src_embed_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

        # Decoder: uses tgt_embed_dim
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=tgt_embed_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True),
            num_layers=num_decoder_layers
        )
        self.memory_proj = nn.Linear(src_embed_dim, tgt_embed_dim)
        self.decoder_input_proj = nn.Linear(tgt_embed_dim, tgt_embed_dim) 
        self.output_layer = nn.Linear(tgt_embed_dim, tgt_vocab_size)

    def forward(self, src_input_ids, tgt_embeddings, src_padding_mask=None, tgt_padding_mask=None, tgt_mask=None):
        # src_input_ids: [batch, src_seq_len]
        # tgt_embeddings: [batch, tgt_seq_len, 768] — from IndicBERT
        src_emb = self.pos_encoder(self.src_embedding(src_input_ids))
        tgt_emb = self.decoder_input_proj(tgt_embeddings)
        tgt_emb = self.pos_decoder(tgt_emb)

        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        memory = self.memory_proj(memory)  # Project encoder output to tgt_embed_di


        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)

        return self.output_layer(output)





from tqdm import tqdm
import torch

def train(model, train_loader, val_loader, optimizer, loss_fn, device, indic_model, num_epochs=2, save_path="fin_model.pth"):
    best_val_loss = float('inf')
    indic_model = indic_model.to(device)


    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)

        for src_ids, tgt_ids, tgt_attention_mask in train_bar:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            tgt_attention_mask = tgt_attention_mask.to(device)

            # Decoder input and target labels
            tgt_input = tgt_ids[:, :-1]
            tgt_input = tgt_input.to(device)
            tgt_labels = tgt_ids[:, 1:]

            with torch.no_grad():
                indic_out = indic_model(input_ids=tgt_input, attention_mask=tgt_attention_mask[:, :-1])
                tgt_embeddings = indic_out.last_hidden_state

            output = model(src_ids, tgt_embeddings, src_padding_mask=(src_ids == 0), tgt_padding_mask=(tgt_input == 0))
            #print("Unique tgt_labels:", torch.unique(tgt_labels))
            output = output.view(-1, output.shape[-1])
            tgt_labels = tgt_labels.contiguous().view(-1)
            assert output.shape[0] == tgt_labels.shape[0]  
            loss = loss_fn(output, tgt_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # ===================== VALIDATION =====================
        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
        
# Move inputs to the same device
        
        

        with torch.no_grad():
            for src_ids, tgt_ids, tgt_attention_mask in val_bar:
                src_ids = src_ids.to(device)
                tgt_ids = tgt_ids.to(device)
                tgt_attention_mask = tgt_attention_mask.to(device)

                tgt_input = tgt_ids[:, :-1]
                tgt_labels = tgt_ids[:, 1:]

                indic_out = indic_model(input_ids=tgt_input, attention_mask=tgt_attention_mask[:, :-1])
                tgt_embeddings = indic_out.last_hidden_state

                output = model(src_ids, tgt_embeddings, src_padding_mask=(src_ids == 0), tgt_padding_mask=(tgt_input == 0))
                pred_ids = output.argmax(dim=-1)

                src_tokens = src_ids[0].tolist()  # Take the first item in the batch
                src_words = [idx2word.get(idx, "<unk>") for idx in src_tokens if idx != vocab["<pad>"]]
                src_sentence = ' '.join(src_words)
                print("English source:", src_sentence)

                print("Predicted token IDs:", pred_ids[0])
                print("Decoded prediction:", indic_tokenizer.decode(pred_ids[0].tolist(), skip_special_tokens=True))
                output = output.view(-1, output.shape[-1])
                tgt_labels = tgt_labels.contiguous().view(-1)
                val_loss = loss_fn(output, tgt_labels)

                total_val_loss += val_loss.item()
                val_bar.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\nEpoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")






import torch.nn as nn

# Assuming padding token id = 0 for Tamil (common choice)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

model = TransformerNMT(
    src_vocab_size=len(eng_vocab),
    tgt_vocab_size=indic_tokenizer.vocab_size,
    src_embedding_matrix=embedding_tensor,
    src_embed_dim=200,
    tgt_embed_dim=768,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)





train(model, train_loader, val_loader, optimizer, loss_fn, device, indic_model, num_epochs=18)
df_infer=pd.read_csv("team18_ta_test.csv")

valid_dataset = TranslationDataset(
    df=df_infer,
    eng_vocab=eng_vocab,
    glove_tokenizer=glove_tokenizer,
    max_len=64  # or whatever fits your model
)

val_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

model.load_state_dict(torch.load("fin_model.pth", map_location=device))
model.to(device)
indic_model = indic_model.to(device)
model.eval()

import csv

from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

references = []
predictions = []

for src_ids, tgt_ids, tgt_attention_mask in tqdm(val_loader, desc="Evaluating"):
    src_ids = src_ids.to(device)
    tgt_ids = tgt_ids.to(device)
    tgt_attention_mask = tgt_attention_mask.to(device)

    tgt_input = tgt_ids[:, :-1]
    tgt_labels = tgt_ids[:, 1:]

    indic_out = indic_model(input_ids=tgt_input, attention_mask=tgt_attention_mask[:, :-1])
    tgt_embeddings = indic_out.last_hidden_state

    output = model(src_ids, tgt_embeddings, src_padding_mask=(src_ids == 0), tgt_padding_mask=(tgt_input == 0))
    pred_ids = output.argmax(dim=-1)

    for i in range(src_ids.size(0)):  # iterate through the batch
        src_tokens = src_ids[i].tolist()
        src_words = [idx2word.get(idx, "<unk>") for idx in src_tokens if idx != vocab["<pad>"]]
        src_sentence = ' '.join(src_words)

        decoded_pred = indic_tokenizer.decode(pred_ids[i].tolist(), skip_special_tokens=True)
        decoded_ref = indic_tokenizer.decode(tgt_labels[i].tolist(), skip_special_tokens=True)


        references.append([decoded_ref.split()])
        predictions.append(decoded_pred.split())

# Compute corpus BLEU
smooth_fn = SmoothingFunction().method4
bleu_score4 = corpus_bleu(references, predictions, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth_fn)
bleu_score3 = corpus_bleu(references, predictions, weights=(0.3333, 0.3333, 0.3333), smoothing_function=smooth_fn)
bleu_score2 = corpus_bleu(references, predictions, weights=(0.5,0.5), smoothing_function=smooth_fn)
bleu_score1 = corpus_bleu(references, predictions, weights=(1.0,), smoothing_function=smooth_fn)

print(f"\nCorpus BLEU-4 Score: {bleu_score4:.4f}")
print(f"\nCorpus BLEU-3 Score: {bleu_score3:.4f}")
print(f"\nCorpus BLEU-2 Score: {bleu_score2:.4f}")
print(f"\nCorpus BLEU-1 Score: {bleu_score1:.4f}")


       


