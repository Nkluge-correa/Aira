import json
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
from dash.dependencies import Input, Output, State
from dash import dcc, html, Output, Input, State
from tensorflow import keras
import dash_bootstrap_components as dbc
import tensorflow as tf
import numpy as np
import unidecode
import string
import time
import dash
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


with open('answers_en.txt', encoding='utf8') as file_in:
    answers = []
    for line in file_in:
        answers.append(line.strip())

model = keras.models.load_model('pre_trained_aira\pre_trained_aira_lstm.h5')

with open('pre_trained_aira\\aira_tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    word_index = tokenizer.word_index


def textbox(text, box='other'):
    style = {
        'max-width': '55%',
        'width': 'max-content',
        'padding': '10px 15px',
        'border-radius': '25px',
    }

    if box == 'self':
        style['margin-left'] = 'auto'
        style['margin-right'] = 0

        color = 'primary'
        inverse = True

    elif box == 'other':
        style['margin-left'] = 0
        style['margin-right'] = 'auto'

        color = 'light'
        inverse = False

    else:
        raise ValueError('Incorrect option for `box`.')

    return dbc.Card(text, style=style, body=True, color=color, inverse=inverse)


conversation = html.Div(
    style={
        'width': '80%',
        'max-width': '1200px',
        'height': '65vh',
        'margin': 'auto',
        'overflow-y': 'auto',
    },
    id='display-conversation',
)

controls = dbc.InputGroup(
    style={'width': '80%', 'max-width': '1200px', 'margin': 'auto'},
    children=[
        dbc.Input(id='user-input', placeholder='Write to Ai.ra...', type='text'),
        dbc.InputGroup(dbc.Button('Submit', size='lg', id='submit')),
    ],
)

modal = html.Div(
    [
        dbc.Button(
            'Information', id='open-body-scroll', outline=True, size='lg', color='primary', n_clicks=0
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(dcc.Markdown(
                    '### What is Ai.ra? 🤔'), style={})),
                dbc.ModalBody([
                    dcc.Markdown("`Ai.ra` is a **chatbot** (or chatterbot). We can also say that Ai.ra is a **language model**, i.e. it is a software application capable of manipulating text. Ai.ra is designed to simulate the way a **human (expert)** would behave during a round of questions and answers (**Q&A**).", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                    'font-size': 20,
                                                                                                                                                                                                                                                                                                                                    'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("We can classify this type of system (**CUS - Conversation Understanding System**) into '*open domain systems*' and '*closed domain systems*'. A closed domain system, also known as a domain-specific system, focuses on a particular set of topics and has limited responses. On the other hand, an open domain system encompasses (in principle) any topic. For example, **GPT-3** - the NLP model produced by OpenAI - **is capable of 'chatting about virtually anything.'**", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("**Ai.ra is a closed domain chatbot**, so don't even try to ask it what the square root of 25 is, it won't be able to help you (but its calculator can!). Ai.ra is designed to provide definitions and answer questions on topics related to `artificial intelligence (AI)`, `machine learning`, `AI ethics`, and `AI safety`, and this is her '*domain*'.", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                     'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                     'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("Ai.ra has **three iterations**, the first and second iterations were trained by machine learning (it used a `Bayesian neural network` and a `Bi-directional LSTM` trained through `supervised learning`), while the third iteration was created from `pre-set rules` (n-gram analysis + dictionary search).", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                       'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                       'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("Ai.ra was developed by [Nicholas Kluge](https://nkluge-correa.github.io/) and [Carolina Del Pino](http://lattes.cnpq.br/6291330432531578). For more information visit [this repository](https://github.com/Nkluge-correa/Aira-EXPERT). ", style={'text-align': 'justify',
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


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

server = app.server
app.title = 'Ai.ra - the AIRES Expert 🤖'

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1('Ai.ra - AIRES artificial expert  🤖', style={'color': '#2a9fd6',
                                                             'font-style': 'bold',
                                                             'margin-top': '15px',
                                                             'margin-left': '15px',
                                                             'display': 'inline-block'}),
        html.Div([modal], style={
                 'display': 'inline-block', 'float': 'right', 'margin-top': '25px'}),
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
        text = tokenizer.texts_to_sequences([texto])
        padded_text = pad_sequences(text, maxlen=100, truncating='post')

        prediction = model.predict(padded_text, verbose=0)
        l = list(prediction[0])
        max_value = max(l)
        index = l.index(max_value)
        bot_input_ids = answers[index-1]
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
