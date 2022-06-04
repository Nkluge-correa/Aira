# ----------------------------------------------------------------------------------------#
# Bibliotecas
# ----------------------------------------------------------------------------------------#
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import dash
import time
import string
import unidecode
import itertools
import statistics
from statistics import mode
import dash_bootstrap_components as dbc
from dash import dcc, html, Output, Input, State
from dash.dependencies import Input, Output, State



# ----------------------------------------------------------------------------------------#
# Chatbot (Aira)
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

sick = dict(zip(questions, labels))

sick.keys()

def generate_ngrams(text, WordsToCombine):
     words = text.split()
     output = []  
     for i in range(len(words)- WordsToCombine+1):
         output.append(words[i:i+WordsToCombine])
     return output

def make_keys(text, WordsToCombine):
    gram = generate_ngrams(text, WordsToCombine)
    sentences = []
    for i in range(0, len(gram)):
        sentence = ' '.join(gram[i])
        sentences.append(sentence)
    return sentences  


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
        'height': '55vh',
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
                dbc.ModalHeader(dbc.ModalTitle(dcc.Markdown('### **O que é a Ai.ra?**'), style={'font-family':'Open Sans'})),
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
            size= 'lg',
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

# ----------------------------------------------------------------------------------------#
# Dash
# ----------------------------------------------------------------------------------------#
app = dash.Dash(__name__, 
            meta_tags=[{'name': 'viewport','content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}],
            external_stylesheets=[dbc.themes.CYBORG])

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
    [
        Input('user-input', 'n_submit'), 
        Input('store-conversation', 'data')]
)
def update_display(user_input, chat_history):
    time.sleep(2)
    if user_input is None or user_input == '':
        return textbox(chat_history, box='other')
    else:
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
        chat_history.append(answers[-1])
        return chat_history, ''


    if user_input is None or user_input == '':
        chat_history = []
        chat_history.append(answers[-1])
        return chat_history, ''

    sentences = []
    values=[]
    text = user_input
    new_text = text.translate(str.maketrans('', '', string.punctuation))
    new_text = new_text.lower()
    new_text = unidecode.unidecode(new_text)

    if len(new_text.split()) == 1:
        if new_text in sick.keys():
                l = [sick[new_text]] * 100
                values.append(l)
        new_text = new_text + ' ' + new_text

    elif len(new_text.split()) != 1:
        if new_text in sick.keys():
                l = [sick[new_text]] * 100
                values.append(l)           
    else:
        pass

    for i in range(1, len(new_text.split()) + 1):
        sentence = make_keys(new_text, i)
        sentences.append(sentence)

    for i in range(0, len(sentences)):
        attention = sentences[i]
        for i in range(0, len(attention)):
            if attention[i] in sick.keys():
                l = [sick[attention[i]]] * i
                values.append(l)
    
    if len(sentences) == 0:
        bot_input_ids = answers[-7]
    if len(values) == 0:
        bot_input_ids = answers[-7]
    else:
        values = list(itertools.chain(*values))
        prediction = mode(values)
        bot_input_ids = answers[int(prediction)-1]
    
    chat_history = []
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