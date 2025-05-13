# clive

1. Perform tokenization (Whitespace, Punctuation-based, Treebank, Tweet, MWE)
using NLTK library. Use porter stemmer and snowball stemmer for stemming. Use
any technique for lemmatization.

import nltk
from nltk.tokenize import WhitespaceTokenizer, word_tokenize, TreebankWordTokenizer, TweetTokenizer
from nltk.tokenize.mwe import MWETokenizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

text=input("Enter your text: ")
ws_tokenizer = WhitespaceTokenizer()
print("Whitespace", ws_tokenizer.tokenize(text))
print("Punctuation based:", word_tokenize(text))

treebank=TreebankWordTokenizer()
print("Treebank:", treebank.tokenize(text))

print("Tweet:", TweetTokenizer().tokenize(text))
mwe_tokenizer=MWETokenizer([('Machine','Learning')])
mwe_tokenizer.add_mwe(('Deep','Learning'))
tokens=word_tokenize(text)
print("MWE:", mwe_tokenizer.tokenize(tokens))

porter=PorterStemmer()
snowball=SnowballStemmer("english")

print("Porter Stemmer:", [porter.stem(token) for token in tokens])
print("Snowball Stemmer:", [snowball.stem(token) for token in tokens])

a=WordNetLemmatizer()
print("Lemmatized:", [a.lemmatize(token) for token in tokens])

2. 2. Perform bag-of-words approach (count occurrence, normalized count occurrence),
TF-IDF on data. Create embeddings using Word2Vec

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
nltk.download('punkt')
nltk.download('punkt_tab')
document = [
    "NLP is facsinating.",
    "Machine Learning includes NLP",
    "Word embeddings help in NLP tasks."
]
vectorizer=CountVectorizer()
bow_matrix=vectorizer.fit_transform(document)
print("\nBag of Words")
print(vectorizer.get_feature_names_out())
print(bow_matrix.toarray())

normalized_bow=bow_matrix.toarray().astype(float)
normalized_bow=normalized_bow/normalized_bow.sum(axis=1,keepdims=True)
print("\nBag of Words: Normaized Count")
print(normalized_bow)

tfidf=TfidfVectorizer()
tfidf_matrix=tfidf.fit_transform(document)
print("\nTF_IDF")
print(tfidf.get_feature_names_out())
print(tfidf_matrix.toarray())

tokenized_docs=[word_tokenize(doc.lower()) for doc in document]
model=Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=2)
print("\nWord2Vec")
print(model.wv.key_to_index.keys())

print("\nVector for 'nlp':")
print(model.wv['nlp'])

print("\nMost similar word to 'nlp':")
print(model.wv.most_similar('nlp'))

3. Perform text cleaning, perform lemmatization (any method), remove stop words
(any method), label encoding. Create representations using TF-IDF. Save outputs.

import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data = {
    'text': [
        "NLP is the future of AI!",
        "Machine Learning helps automate tasks.",
        "Lemmatization reduces words to their base form.",
        "Stopwords should be removed before vectorization.",
        "AI and ML are powerful tools."
    ],
    'label': ['tech', 'tech', 'nlp', 'nlp', 'tech']
}

df=pd.DataFrame(data)
print(df.head())

def clean_text(text):
  text=text.lower()
  text=re.sub(r'[^a-z\s]', '', text)
  return text

df['clean_text']=df['text'].apply(clean_text)

lemmatizer=WordNetLemmatizer()
def lemmatize_text(text):
  tokens=nltk.word_tokenize(text)
  lemmatized=[lemmatizer.lemmatize(token) for token in tokens]
  return ' '.join(lemmatized)

df['lemmatized_text']=df['clean_text'].apply(lemmatize_text)

stop_words=set(stopwords.words('english'))
def remove_stopwords(text):
  tokens=nltk.word_tokenize(text)
  filtered=[word for word in tokens if word not in stop_words]
  return ' '.join(filtered)

df['final_text']=df['lemmatized_text'].apply(remove_stopwords)

label_encoder=LabelEncoder()
df['encoded_label']=label_encoder.fit_transform(df['label'])

vectorizer=TfidfVectorizer()
tfidf_matrix=vectorizer.fit_transform(df['final_text'])
tfidf_df=pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

output=pd.concat([df[['text','label','final_text','encoded_label']],tfidf_df], axis=1)
output.to_csv("output.csv", index=False)
print("Outputs Saved")

4. Create a transformer from scratch using the Pytorch library

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

#scaled_dot_product_attention
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.linear_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linear_out(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(heads, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ff_out = self.feed_forward(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1, max_len=512):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, src, mask=None):
        x = self.embedding(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

src = torch.randint(0, 1000, (2, 6))  # batch_size x seq_len

model = TransformerEncoder(
    vocab_size=1000,
    d_model=512,
    N=2,
    heads=8,
    d_ff=2048,
    dropout=0.1,
    max_len=100
)
output = model(src)
print(output.shape)  # Should be (2, 6, 512)

5. Perform different parsing techniques using Shallow parser, regex parser

import nltk
from nltk import word_tokenize, pos_tag, RegexpParser

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

sentence=input("\nEnter a sentence: ")
tokens = word_tokenize(sentence)
tagged_tokens=pos_tag(tokens)
print("Pos Tagged: ", tagged_tokens)

chunk_grammar= r"""
  NP: {<DT>?<JJ>*<NN>}
  VP: {<VB.*><IN>}
  """
chunk_parser=RegexpParser(chunk_grammar)
tree=chunk_parser.parse(tagged_tokens)
print("Shallow Parsing Tree: ")
print(tree)
tree.pretty_print()

regex_grammar=r"""
  ADJP: {<JJ><NN>}
  ACTION:{<NN><VB.*>}
  """

regex_parser=RegexpParser(regex_grammar)
regex_tree=regex_parser.parse(tagged_tokens)
print("Regex Parsing Tree:")
print(regex_tree)
regex_tree.pretty_print()

6. A) Apply log linear model for sentiment analysis

import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

nltk.download('movie_reviews')
nltk.download('punkt')

docs = []
labels = []

for category in movie_reviews.categories():
    for fid in movie_reviews.fileids(category):
        docs.append(movie_reviews.raw(fid))
        labels.append(category)

docs_train, docs_test, y_train, y_test = train_test_split(
    docs, labels, test_size=0.2, random_state=42, stratify=labels
)

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
    ('clf', LogisticRegression(solver='lbfgs', max_iter=1000)),
])

pipe.fit(docs_train, y_train)

y_pred = pipe.predict(docs_test)

print("\n=== Sentiment Analysis Results ===")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred, digits=4))

6. B) B)Implement and extract Named Entity recognition techniques using given text:
&#39;&#39;&#39;Deepak Jasani, Head of retail research, HDFC Securities, said: “Investors will look
to the European Central Bank later Thursday for reassurance that surging prices are
just transitory, and not about to spiral out of control. In addition to the ECB policy
meeting, investors are awaiting a report later Thursday on US economic growth,
which is likely to show a cooling recovery, as well as weekly jobs data.”.&#39;&#39;&#39;

import spacy
nlp=spacy.load('en_core_web_sm')
text = (
    "Deepak Jasani, Head of retail research, HDFC Securities, said: "
    "“Investors will look to the European Central Bank later Thursday for reassurance "
    "that surging prices are just transitory, and not about to spiral out of control. "
    "In addition to the ECB policy meeting, investors are awaiting a report later Thursday "
    "on US economic growth, which is likely to show a cooling recovery, as well as weekly jobs data.”"
)
doc = nlp(text)
print("Named Entities:")
for ent in doc.ents:
  print(f"{ent.text:30} → {ent.label_}")

7. 7.A) Implementing Non-Negative Matrix Factorization (NMF) for topic modeling and
evaluate using the reconstruction error.

import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

newsgroups=fetch_20newsgroups(remove=('headers','footers','quotes'))
documents=newsgroups.data[:1000]

vectroizer=TfidfVectorizer(stop_words=stopwords.words('english'), max_features=2000)
tfidf=vectorizer.fit_transform(documents)

num_topics=5
nmf_model=NMF(n_components=num_topics, random_state=42)
W=nmf_model.fit_transform(tfidf)
H=nmf_model.components_

feature_names=vectorizer.get_feature_names_out()
print("Topics Discovered")
for topic_idx, topic in enumerate(H):
  top_words=[feature_names[i] for i in topic.argsort()[:-11:-1]]
  print(f"Topic #{topic_idx + 1}: {' | '.join(top_words)}")

reconstructed=np.dot(W,H)
reconstructed_error=mean_squared_error(tfidf.toarray(), reconstructed)
print("\nReconstruction Error: ", reconstructed_error)

7. B) Implement Wordnet to show Word Disambiguition.

import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

sentence="He deposited money in the bank"
word="bank"

context = word_tokenize(sentence)
synset=lesk(context, word)

print("Word Sense Disambiguation")
print(f"Word: {word}")
print("Synset:", synset)
print("Definition:", synset.definition() if synset else "No definition found.")

8. Not here for now

9. A) Implement n-gram model.

import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
nltk.download('punkt')

text = [
    'the quick brown fox',
    'the slow brown dog',
    'the quick red dog',
    'the lazy yellow fox'
]

def generate_ngrams(texts, n):
    for sentence in texts:
        tokens = word_tokenize(sentence.lower())
        n_grams = list(ngrams(tokens, n))
        print(f"\n{n}-grams for: \"{sentence}\"")
        for gram in n_grams:
            print(gram)

generate_ngrams(text, 1)  # Unigrams
generate_ngrams(text, 2)  # Bigrams
generate_ngrams(text, 3)  # Trigrams

9. B) Implement Latent Dirichilet Allocation model and Latent Semantic Analysis for
topic modelling using text: &#39;the quick brown fox&#39;,
       &#39;the slow brown dog&#39;,
       &#39;the quick red dog&#39;,
       &#39;the lazy yellow fox &#39;

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

documents = [
    "The cat sat on the mat.",
    "Dogs are great pets.",
    "I love to play football.",
    "Data science is an interdisciplinary field.",
    "Python is a great programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Artificial intelligence and machine learning are popular topics.",
    "Deep learning is a type of machine learning.",
    "Natural language processing involves analyzing text data.",
    "I enjoy hiking and outdoor activities."
]

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
lda_model.fit(X)

terms = vectorizer.get_feature_names_out()

print("\n🔹 LDA Topics Discovered:")
for idx, topic in enumerate(lda_model.components_):
    print(f"\nTopic #{idx + 1}:")
    print(" | ".join([terms[i] for i in topic.argsort()[-7:][::-1]]))

10. Fine tune a pre-trained transformer for any of the following tasks on any relevant
dataset of your choice:  1. Neural Machine Translation  2. Classification 

from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

dataset = load_dataset("imdb")
dataset = dataset.shuffle(seed=42)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)
dataset = dataset.map(tokenize, batched=True)

dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(10000)),  # Reduce for quick training
    eval_dataset=dataset["test"].select(range(2000)),
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate()
print("\n📊 Evaluation Results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
