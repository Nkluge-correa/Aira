# ----------------------------------------------------------------------------------------#
# Bibliotecas
# ----------------------------------------------------------------------------------------#
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import json 
import dash
import time
import string
import unidecode
import pandas as pd
import numpy as np
import tensorflow as tf
import dash_bootstrap_components as dbc
from tensorflow import keras
from keras import regularizers 
from dash import dcc, html, Output, Input, State
from dash.dependencies import Input, Output, State
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json

# ----------------------------------------------------------------------------------------#
# Load Model & Tokenizer
# ----------------------------------------------------------------------------------------#
with open('answers.txt', encoding='utf8') as file_in:
    answers = []
    for line in file_in:
        answers.append(line.strip())

model = keras.models.load_model('pre_trained_aira_lstm')

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    word_index = tokenizer.word_index
# ----------------------------------------------------------------------------------------#
# Dash Back-end
# ----------------------------------------------------------------------------------------#

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
        'max-width': '800px',
        'height': '65vh',
        'margin': 'auto',
        'overflow-y': 'auto',
    },
    id='display-conversation',
)

controls = dbc.InputGroup(
    style={'width': '80%', 'max-width': '800px', 'margin': 'auto'},
    children=[
        dbc.Input(id='user-input', placeholder='Escreva para Ai.ra...', type='text'),
        dbc.InputGroup(dbc.Button('Enviar', size= 'lg', id='submit')),
    ],
)

modal = html.Div(
    [
        dbc.Button(
            'Informações', id='open-body-scroll', outline=True, size= 'lg', color='primary', n_clicks=0
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle('O que é a Ai.ra?')),
                dbc.ModalBody([dcc.Markdown(answers[-6]), html.Br(),
                                dcc.Markdown(answers[-5]), html.Br(),
                                dcc.Markdown(answers[-4]), html.Br(),
                                dcc.Markdown(answers[-3]), html.Br(),
                                dcc.Markdown(answers[-2]), 
                ], style={'font-family':'Open Sans',
                            'text-align': 'justify',
                            'font-size': 20,
                            'text-justify': 'inter-word'}),
                dbc.ModalFooter(
                    dbc.Button(
                        'Fechar',
                        id='close-body-scroll',
                        className='ms-auto',
                        n_clicks=0,
                        size= 'lg'
                    )
                ),
            ],
            id='modal-body-scroll',
            scrollable=True,
            is_open=False,
        ),
    ], style={'position': 'relative',
            'left':'600px',
            'bottom':'10px',
            'display':'inline-block'},

)

def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# ----------------------------------------------------------------------------------------#
# Dash
# ----------------------------------------------------------------------------------------#
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

server = app.server
app.title = 'Ai.ra - the AIRES Expert'
# ----------------------------------------------------------------------------------------#
# Layout
# ----------------------------------------------------------------------------------------#

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1('Ai.ra - a expert artificial da AIRES  ', style={'color':'#2a9fd6',
                                                            'font-style': 'bold', 
                                                            'margin-top': '15px',
                                                            'margin-left': '15px',
                                                            'display':'inline-block'}),
        html.H1('  🤖', style={'color':'#2a9fd6',
                            'font-style': 'bold', 
                            'margin-top': '15px',
                            'margin-left': '15px',
                            'display':'inline-block'}),
        modal,
        html.Hr(),
        dcc.Store(id='store-conversation', data=''),
        dcc.Loading(id='loading_0', type='circle', children=[conversation]),
        controls,
    ],
)

# ----------------------------------------------------------------------------------------#
# Funçoes
# ----------------------------------------------------------------------------------------#


@app.callback(
    Output('display-conversation', 'children'), 
    [Input('store-conversation', 'data')]

)
def update_display(chat_history):
    time.sleep(2)
    return [
        textbox(chat_history, box='self') if i % 2 == 0 else textbox(chat_history, box='other')
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
    if n_clicks == 0:
        chat_history = []
        chat_history.append('👋🤖')
        chat_history.append(answers[-1])
        return chat_history, ''


    if user_input is None or user_input == '':
        chat_history = []
        chat_history.append('👋🤖')
        chat_history.append(answers[-1])
        return chat_history, ''
    
    else:
    
        texto = user_input
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        texto = texto.lower()
        texto = unidecode.unidecode(texto)
        texto = [texto]
        text = tokenizer.texts_to_sequences(texto)
        padded_text = keras.preprocessing.sequence.pad_sequences(text,
                                                                value=word_index["<PAD>"],
                                                                padding='post',
                                                                maxlen=20)

        prediction = model.predict(padded_text)
        l = list(prediction[0])
        max_value = max(l)
        index = l.index(max_value)
        max_value = '{:.2f}'.format(max(l)*100)
        bot_input_ids = answers[index-1]
        acc = (f'Probabilidade: {max_value}%')
        chat_history = []

        chat_history.append(user_input)
        chat_history.append(bot_input_ids)
        chat_history.append(acc)


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

