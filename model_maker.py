# ----------------------------------------------------------------------------------------#
# Bibliotecas
# ----------------------------------------------------------------------------------------#
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import io 
import json 
import string
import unidecode
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from IPython.core.display import display, HTML
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM

# ----------------------------------------------------------------------------------------#
# Load Data
# ----------------------------------------------------------------------------------------#
with open('tags.txt', encoding='utf8') as file_in:
    questions = []
    labels = []
    for line in file_in:
        labels.append(line.strip().split(' ')[-1])
        questions.append(' '.join(line.strip().split(' ')[:-1]))

with open('answers.txt', encoding='utf8') as file_in:
    answers = []
    for line in file_in:
        answers.append(line.strip())

# ----------------------------------------------------------------------------------------#
# Keras tokenization
# ----------------------------------------------------------------------------------------#
maxlen = 20
training_samples = 1800
validation_samples = 142
max_words = 800 #Unique tokens in questions

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(questions)
sequences = tokenizer.texts_to_sequences(questions)
word_index = tokenizer.word_index
word_index["<PAD>"] = 792  
print('Encontrados %s tokens únicos.' % len(word_index))
padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=maxlen)

tokenizer_json = tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

# ----------------------------------------------------------------------------------------#
# Shuffle & Slip
# ----------------------------------------------------------------------------------------#

labels = np.asarray(labels)
labels = labels.astype(np.float)
labelos = pd.DataFrame(labels)
labelos = labelos.rename(columns={0: 'label'})
labelos = pd.get_dummies(labelos['label'], prefix='label_')
labels = labelos.to_numpy()

print('Formato do tensor (data):', padded_sequences.shape)
print('Formato do tensor (labels)', labels.shape)

indices = np.arange(padded_sequences.shape[0])
np.random.shuffle(indices)
padded_sequences = padded_sequences[indices]
labels = labels[indices]
x_train = padded_sequences[:training_samples]
y_train = labels[:training_samples]
x_val = padded_sequences[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# ----------------------------------------------------------------------------------------#
# Word Embeddings - Repositório de Word Embeddings do NILC (Núcleo Interinstitucional de Linguística Computacional)
# cbow_s50.txt = http://143.107.183.175:22980/download.php?file=embeddings/wang2vec/cbow_s50.zip
# ----------------------------------------------------------------------------------------#

with open('cbow_s50.txt', encoding='utf8') as file_in:
    embeddings_index = {}
    for line in file_in:
        values = line.split()
        word = values[0]
        if isinstance(values[1], str) is True:
            values[1] = values[1].replace(values[1],'0.0')
            values[1] = float(values[1])
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 50
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# ----------------------------------------------------------------------------------------#
# Model
# ----------------------------------------------------------------------------------------#

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(LSTM(100))
model.add(Dense(142, activation='softmax'))
model.summary()


model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False #untrainable frozen layer (NILC - word vectors)

model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])
history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=5,
                    validation_data=(x_val, y_val))

model.save('pre_trained_aira')

# ----------------------------------------------------------------------------------------#
# Logs & Plot
# ----------------------------------------------------------------------------------------#
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

fig = go.Figure(layout={'template':'plotly_dark'})

fig.add_trace(go.Scatter(x=list(epochs), y=acc,
                         line_color='rgba(0, 102, 255, 0.5)', line=dict(width=3, dash='dash'), name='Acurácia (Treinamento)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Acurácia (Treinamento): %{y:.5f} acc <extra></extra>',
                         showlegend=True))
fig.add_trace(go.Scatter(x=list(epochs), y=val_acc,
                         line_color='rgba(255, 0, 0, 0.5)', line=dict(width=3, dash='dash'), name='Acurácia (Validação)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Acurácia (Validação): %{y:.2f} acc <extra></extra>',
                         showlegend=True))

fig.update_xaxes(showgrid=False, showline=False, mirror=False)
fig.update_yaxes(showgrid=True, ticksuffix=' acc')
fig.update_layout(
    paper_bgcolor='#242424',
    plot_bgcolor='#242424',
    hovermode='x unified',
    font_family='Open Sans',
    autosize=True,
    margin=dict(l=10, r=10, b=10, t=10),
    hoverlabel=dict(bgcolor='#242424', font_size=18, font_family='Open Sans')
)

fig.show()

fig2 = go.Figure(layout={'template':'plotly_dark'})

fig2.add_trace(go.Scatter(x=list(epochs), y=loss,
                         line_color='rgba(0, 102, 255, 0.5)', line=dict(width=3, dash='dash'), name='Loss (Treinamento)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Loss (Treinamento): %{y:.5f} loss <extra></extra>',
                         showlegend=True))
fig2.add_trace(go.Scatter(x=list(epochs), y=val_loss,
                         line_color='rgba(255, 0, 0, 0.5)', line=dict(width=3, dash='dash'), name='Loss (Validação)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Loss (Validação): %{y:.2f} loss <extra></extra>',
                         showlegend=True))

fig2.update_xaxes(showgrid=False, showline=False, mirror=False)
fig2.update_yaxes(showgrid=True, ticksuffix=' loss')
fig2.update_layout(
    paper_bgcolor='#242424',
    plot_bgcolor='#242424',
    hovermode='x unified',
    font_family='Open Sans',
    autosize=True,
    margin=dict(l=10, r=10, b=10, t=10),
    hoverlabel=dict(bgcolor='#242424', font_size=18, font_family='Open Sans')
)

fig2.show()