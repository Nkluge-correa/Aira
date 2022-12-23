from dash.dependencies import Input, Output, State
from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
from statistics import mode
import itertools
import unidecode
import string
import json
import time
import dash

with open('data\\tags_en.txt', encoding='utf8') as file_in:
    X = [' '.join(line.strip().split(' ')[:-1]) for line in file_in]
with open('data\\tags_en.txt', encoding='utf8') as file_in:
    Y = [line.strip().split(' ')[-1] for line in file_in]

vocabulary = dict(zip(X, Y))

# english version
with open('data\\answers_en.txt', encoding='utf8') as file_in:
    answers = [line.strip() for line in file_in]


def generate_ngrams(text, WordsToCombine):
    words = text.split()
    output = []
    for i in range(len(words) - WordsToCombine+1):
        output.append(words[i:i+WordsToCombine])
    return output


def make_keys(text, WordsToCombine):
    gram = generate_ngrams(text, WordsToCombine)
    sentences = []
    for i in range(0, len(gram)):
        sentence = ' '.join(gram[i])
        sentences.append(sentence)
    return sentences


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
                    dcc.Markdown("**`Ai.ra` is a chatbot (or chatterbot). We can also say that Ai.ra is a language model, i.e. it is a software application capable of manipulating text. Ai.ra is designed to simulate the way an expert would behave during a round of questions and answers (Q&A).**", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                 'font-size': 20,
                                                                                                                                                                                                                                                                                                                 'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("**We can classify this type of system (CUS - _Conversation Understanding System_) into '_open domain systems_' and '_closed domain systems_'. A closed domain system, also known as a domain-specific system, focuses on a particular set of topics and has limited responses. On the other hand, an open domain system encompasses (_in principle_) any topic. For example, GPT-3 - the NLP model produced by OpenAI - is capable of 'chatting about virtually anything.'**", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("**Ai.ra is a _closed domain chatbot_, so don't even try to ask it what the square root of 25 is. It won't be able to help you (but your calculator can!). Ai.ra is designed to provide definitions and answer questions on topics related to `artificial intelligence (AI)`, `machine learning`, `AI ethics`, and `AI safety`, and this is her '_domain*_'.**", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                                       'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                                       'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("**Ai.ra has **four iterations**, the first and second iterations were trained by machine learning (a `Bayesian neural network`, a `Bi-directional LSTM`, and a `Decoder-Transformer` were trained through `supervised learning`), while the third iteration was created from `pre-set rules` (n-gram analysis + dictionary search).`**", style={'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                  'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                  'text-justify': 'inter-word'}), html.Br(),
                    dcc.Markdown("**Ai.ra was developed by [Nicholas Kluge](https://nkluge-correa.github.io/) and [Carolina Del Pino](http://lattes.cnpq.br/6291330432531578). For more information visit [this repository](https://github.com/Nkluge-correa/Aira-EXPERT).**", style={'text-align': 'justify',
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


app = dash.Dash(__name__,
                meta_tags=[
                    {'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}],
                external_stylesheets=[dbc.themes.CYBORG])

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
        sentences = []
        values = []
        text = user_input
        new_text = text.translate(str.maketrans('', '', string.punctuation))
        new_text = new_text.lower()
        new_text = unidecode.unidecode(new_text)

        if len(new_text.split()) == 1:
            if new_text in vocabulary.keys():
                l = [vocabulary[new_text]] * 100
                values.append(l)
            new_text = new_text + ' ' + new_text

        elif len(new_text.split()) != 1:
            if new_text in vocabulary.keys():
                l = [vocabulary[new_text]] * 100
                values.append(l)
        else:
            pass

        for i in range(1, len(new_text.split()) + 1):
            sentence = make_keys(new_text, i)
            sentences.append(sentence)

        for i in range(0, len(sentences)):
            attention = sentences[i]
            for i in range(0, len(attention)):
                if attention[i] in vocabulary.keys():
                    l = [vocabulary[attention[i]]] * i
                    values.append(l)

        if len(values[0]) == 0:
            chat_history.append(f"'{user_input}'")
            bot_input_ids = answers[-1]
            chat_history.append(bot_input_ids)
            return chat_history, ''

        elif len(values) != 0:
            values = list(itertools.chain(*values))
            prediction = mode(values)
            bot_input_ids = answers[int(prediction)-1]
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
