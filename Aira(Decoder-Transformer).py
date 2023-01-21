from keras_preprocessing.sequence import pad_sequences
from dash import dcc, html, Output, Input, State
from keras.layers import TextVectorization
from tensorflow import keras
from keras import layers
import dash_bootstrap_components as dbc
import tensorflow as tf
import numpy as np
import unidecode
import string
import time
import dash


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config


with open('data\\answers_en.txt', encoding='utf-8') as fp:
    answers = [line.strip() for line in fp]
    fp.close()

with open(r'pre_trained_aira\\vocabulary_bilingual.txt', encoding='utf-8') as fp:
    vocabulary = [line[:-1] for line in fp]
    fp.close()

model = keras.models.load_model("pre_trained_aira\\Aira_transformer_bilingual.keras",
                                custom_objects={"TransformerEncoder": TransformerEncoder,
                                                "PositionalEmbedding": PositionalEmbedding})

text_vectorization = TextVectorization(max_tokens=2500,
                                       output_mode="int",
                                       ngrams=2,
                                       vocabulary=vocabulary)


def textbox(text, box='other'):
    style = {
        'max-width': '55%',
        'width': 'max-content',
        'padding': '10px 15px',
        'border-radius': '25px',
        'margin-bottom': '10px'
    }

    if box == 'self':
        style['margin-left'] = 'auto'
        style['margin-right'] = 0

        color = 'primary'
        inverse = True

    elif box == 'other':
        style['margin-left'] = 0
        style['margin-right'] = 'auto'

        color = 'secondary'
        inverse = True

    else:
        raise ValueError('Incorrect option for `box`.')

    return dbc.Card(text, style=style, body=True, color=color, inverse=inverse)


conversation = html.Div(
    style={
        'width': '80%',
        'max-width': '800px',
        'height': '60vh',
        'margin': 'auto',
        'overflow-y': 'auto',
    },
    id='display-conversation',
)

controls = dbc.InputGroup(
    style={'width': '80%', 'max-width': '800px', 'margin': 'auto'},
    children=[
        dbc.Input(id='user-input', placeholder='Write to Ai.ra...', type='text'),
        dbc.InputGroup(dbc.Button('Submit', size='lg', id='submit')),
    ],
)

modal = html.Div(
    [
        dbc.Button(
            'Information', id='open-body-scroll', outline=True, size='lg', color='light', n_clicks=0
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(dcc.Markdown(
                    '### What is `Ai.ra`? 🤔'), style={})),
                dbc.ModalBody([
                    dcc.Markdown("**`Ai.ra` is a chatbot (or chatterbot). We can also say that Ai.ra is a language model, i.e. it is a software application capable of manipulating text. Ai.ra is designed to simulate the way an expert would behave during a round of questions and answers (Q&A).**", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                 'font-size': 20,
                                                                                                                                                                                                                                                                                                                 'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("**We can classify this type of system (CUS - _Conversation Understanding System_) into '_open domain systems_' and '_closed domain systems_'. A closed domain system, also known as a domain-specific system, focuses on a particular set of topics and has limited responses. On the other hand, an open domain system encompasses (_in principle_) any topic. For example, GPT-3 - the NLP model produced by OpenAI - is capable of 'chatting about virtually anything.'**", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("**`Ai.ra` is a _closed domain chatbot_, so don't even try to ask it what the square root of 25 is. It won't be able to help you (but your calculator can!). Ai.ra is designed to provide definitions and answer questions on topics related to `artificial intelligence (AI)`, `machine learning`, `AI ethics`, and `AI safety`, and this is her '_domain*_'.**", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                           'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                           'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("**`Ai.ra` has **four iterations**, the first and second iterations were trained by machine learning (a `Bayesian neural network`, a `Bi-directional LSTM`, and a `Decoder-Transformer` were trained through `supervised learning`), while the third iteration was created from `pre-set rules` (n-gram analysis + dictionary search).`**", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                    'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                    'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("**`Ai.ra` was developed by [`Nicholas Kluge`](https://nkluge-correa.github.io/) and [`Carolina Del Pino`](http://lattes.cnpq.br/6291330432531578). For more information visit this [`repository`](https://github.com/Nkluge-correa/Aira-EXPERT).**", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                              'font-size': 20,
                                                                                                                                                                                                                                                                                              'text-justify': 'inter-word'}),
                ]),
                dbc.ModalFooter(
                    dbc.Button(
                        'Fechar',
                        id='close-body-scroll',
                        className='ms-auto',
                        n_clicks=0,
                        size='lg'
                    )
                ),
            ],
            id='modal-body-scroll',
            scrollable=True,
            size='lg',
            is_open=False,
        ),
    ], style={
        'margin-top': '5px',
        'margin-left': '15px',
    },

)


def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

server = app.server
app.title = 'Ai.ra - the AIRES Expert 🤖'

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1('Ai.ra - AIRES artificial expert  🤖', style={'textAlign': 'center',
                                                             'margin-top': '20px'}),
        html.Div([modal], style={'textAlign': 'center'}),
        html.Hr(),
        dcc.Store(id='store-conversation', data=''),
        dcc.Loading(id='loading_0', type='circle', children=[conversation]),
        controls,
    ],
)


@app.callback(
    Output('display-conversation', 'children'),
    [Input('store-conversation', 'data')]

)
def update_display(chat_history):
    time.sleep(2)
    return [
        textbox(chat_history, box='self') if i % 2 == 0 else textbox(
            chat_history, box='other')
        for i, chat_history in enumerate(chat_history)
    ]


@app.callback(
    [
        Output('store-conversation', 'data'),
        Output('user-input', 'value')
    ],

    [
        Input('submit', 'n_clicks'),
        Input('user-input', 'n_submit')
    ],

    [
        State('user-input', 'value'),
        State('store-conversation', 'data')
    ]
)
def run_chatbot(n_clicks, n_submit, user_input, chat_history):
    chat_history = chat_history or []
    if n_clicks == 0:
        chat_history.append('👋🤖')
        chat_history.append("Hello, how are you? My name is Ai.ra, but you can call me Ai. I am an artificial intelligence (AI). More specifically, I am an NLP (Natural Language Processing) model trained in conversation (a chatbot!). I have been specifically trained to answer questions about AI Ethics and AI Safety! Would you like a summary of the terms I am aware of?")
        return chat_history, ''

    if user_input is None or user_input == '':
        chat_history.append('👋🤖')
        chat_history.append("Hello, how are you? My name is Ai.ra, but you can call me Ai. I am an artificial intelligence (AI). More specifically, I am an NLP (Natural Language Processing) model trained in conversation (a chatbot!). I have been specifically trained to answer questions about AI Ethics and AI Safety! Would you like a summary of the terms I am aware of?")
        return chat_history, ''

    else:

        texto = user_input
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        texto = texto.lower()
        texto = unidecode.unidecode(texto)
        encoded_sentence = text_vectorization(
            texto.lower().translate(str.maketrans('', '', string.punctuation)))
        encoded_sentence_padded = pad_sequences(
            [encoded_sentence], maxlen=10, truncating='post')

        preds = model.predict(encoded_sentence_padded, verbose=0)[0]
        output = answers[np.argmax(preds)]
        bot_input_ids = f'''{output}
        [Confidence: {max(preds) * 100: .2f} %]'''
        chat_history.append(user_input)
        chat_history.append(bot_input_ids)

        return chat_history, ''


app.callback(
    Output('modal-body-scroll', 'is_open'),
    [
        Input('open-body-scroll', 'n_clicks'),
        Input('close-body-scroll', 'n_clicks'),
    ],
    [State('modal-body-scroll', 'is_open')],
)(toggle_modal)

if __name__ == '__main__':
    app.run_server(debug=False)
