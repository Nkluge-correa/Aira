from sklearn.feature_extraction.text import CountVectorizer
from dash import dcc, html, Output, Input, State
from sklearn.naive_bayes import MultinomialNB
import dash_bootstrap_components as dbc
import numpy as np
import unidecode
import random
import string
import dash

OPEN_BUTTON = 'light'
OUTLINE_BUTTON = True
STYLE_BUTTON = {'border': 0}
CLOSE_BUTTON = 'primary'
FONT_SIZE = '1rem'

avatars = ['🧒', '👧', '🧒🏿', '👱', '👨‍🦱', '👨🏿', '👩‍🦲',
           '🧔', '👩', '👩🏿', '👩‍🦳', '👴', '👱‍♀️', '👨', '👩‍🦰']

avatar = random.choice(avatars)

with open('data/tags_en.txt', encoding='utf-8') as fp:
    questions = [' '.join(line.strip().split(' ')[:-1]) for line in fp]
    fp.close()

with open('data/tags_en.txt', encoding='utf-8') as fp:
    labels = [line.strip().split(' ')[-1] for line in fp]
    fp.close()

with open('data/answers_en.txt', encoding='utf-8') as fp:
    answers = [line.strip() for line in fp]
    fp.close()

bow_vectorizer = CountVectorizer()
training_vectors = bow_vectorizer.fit_transform(questions)
classifier = MultinomialNB()
classifier.fit(training_vectors, labels)


def textbox(text, box='other'):
    style = {
        'max-width': '100%',
        'width': 'max-content',
        'padding': '10px 10px',
        'margin-bottom': '20px',
        'text-align': 'justify',
        'text-justify': 'inter-word',
        'font-size': '1.em',
        'font-weight': 'bold'
    }

    if box == 'self':
        style['float'] = 'right'

    elif box == 'other':
        style['float'] = 'right'

    return dcc.Markdown(text, style=style)


conversation = html.Div(
    style={'width': '100%',
           'max-width': '100vw',
           'height': '70vh',
           'margin': 'auto',
           'overflow': 'auto',
           'display': 'flex',
           'flex-direction': 'column-reverse'},
    id='display-conversation',
)

controls = dbc.InputGroup(
    style={'width': '100%', 'max-width': '100vw',
           'margin': 'auto'},
    children=[
        dbc.Input(id='user-input', placeholder='Write to Ai.ra...',
                  type='text', style={'border-radius': '5px'}),
        dbc.Button(
            [html.I(className="bi bi-send")], size='lg', id='submit',
            outline=True, color='light', style={
                'width': '100%',
                'max-width': '100vw',
                'margin': 'auto',
                'background-color': 'none'}),
    ],
)

modal = html.Div(
    [
        html.A([html.I(className='bi bi-info-circle')],
               id="open-body-scroll", n_clicks=0, className="icon-button", style={'font-size': 25}),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(dcc.Markdown(
                    '## What is `Ai.ra`? 🤔'), style={'font-weight': 'bold'})),
                dbc.ModalBody([
                    dcc.Markdown('''`Ai.ra` is a chatbot (or chatterbot). We can also say that `Ai.ra` is a language model, i.e. it is a software application capable of manipulating text. `Ai.ra` is designed to simulate the way an expert would behave during a round of questions and answers (Q&A).''',
                                 className='modal-body-text-style', style={'font-size': FONT_SIZE}), html.Br(),
                    dcc.Markdown('''We can classify this type of system (CUS - `Conversation Understanding System`) into "_open domain systems_" and "_closed domain systems_". A closed domain system, also known as a domain-specific system, focuses on a particular set of topics and has limited responses. On the other hand, an open domain system could (_in principle_) sustain a dialog about any topic. For example, `GPT-3` - the language model produced by OpenAI - is capable of "chatting about virtually anything."''',
                                 className='modal-body-text-style', style={'font-size': FONT_SIZE}), html.Br(),
                    dcc.Markdown('''`Ai.ra` is a _closed domain chatbot_, so don't even try to ask it what the square root of 25 is. It won't be able to help you (but your calculator can!). `Ai.ra` is designed to provide definitions and answer questions on topics related to `artificial intelligence (AI)`, `machine learning`, `AI ethics`, and `AI safety`, and this is her "_domain_".''',
                                 className='modal-body-text-style', style={'font-size': FONT_SIZE}), html.Br(),
                    dcc.Markdown('''`Ai.ra` has four iterations, the first three were trained as machine learning models (a `Bayesian neural network`, a `Bi-directional LSTM`, and a `Decoder-Transformer` were trained through `supervised learning`), while the fourth iteration was created from `pre-set rules` (n-gram analysis + dictionary search).''',
                                 className='modal-body-text-style', style={'font-size': FONT_SIZE}), html.Br(),
                    dcc.Markdown(
                        '''`Ai.ra` was developed by [`Nicholas Kluge`](https://nkluge-correa.github.io/) and [`Carolina Del Pino`](http://lattes.cnpq.br/6291330432531578). For more information visit this [`repository`](https://github.com/Nkluge-correa/Aira-EXPERT).''', className='modal-body-text-style', style={'font-size': FONT_SIZE}),
                ]),
                dbc.ModalFooter(
                    dbc.Button(
                        html.I(className="bi bi-x-circle"),
                        id='close-body-scroll',
                        className='ms-auto',
                        outline=True,
                        size='xl',
                        n_clicks=0,
                        color=CLOSE_BUTTON,
                        style=STYLE_BUTTON
                    )
                ),
            ],
            id='modal-body-scroll',
            scrollable=True,
            size='xl',
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


badges = html.Span([
    dbc.Badge([html.I(className="bi bi-heart-fill"), "  Open-Source"], href="https://github.com/Nkluge-correa/",
              color="dark", className="text-decoration-none", style={'margin-right': '5px'}),
    dbc.Badge([html.I(className="bi bi-bank"), "  AIRES at PUCRS"], href="https://www.airespucrs.org/",
              color="dark", className="text-decoration-none", style={'margin-right': '5px'}),
    dbc.Badge([html.I(className="bi bi-filetype-py"), "  Made with Python"], href="https://www.python.org/",
              color="dark", className="text-decoration-none", style={'margin-right': '5px'}),
    dbc.Badge([html.I(className="bi bi-github"), "  Nkluge-correa"], href="https://nkluge-correa.github.io/",
              color="dark", className="text-decoration-none", style={'margin-right': '5px'})
])

app = dash.Dash(__name__,
                meta_tags=[
                    {'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}],
                external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])

server = app.server
app.title = 'Ai.ra Expert 🤖'


app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([
            dbc.Col([
                html.Div([dcc.Markdown('# `Ai.ra Expert`', className='title-style'),
                          html.Img(src=dash.get_asset_url(
                              'robot.svg'), height="50px", className='title-icon-style')],
                    className='title-div'),
                html.Div([
                    html.Div([
                        dcc.Markdown('''
                _Ai.ra is a chatbot designed to simulate the way an expert would behave during a round of questions and answers (Q&A). Ai.ra is a small decoder-only transformer model that basically does text classification (classifies your question into one of her domain answers). For comparison reasons, we allow users to toggle between talking to Ai.ra (close-domain) and Distill-Blenderbot (open-domain) on this page._
                ''', className='page-intro')
                    ], className='page-intro-inner-div'),
                ], className='page-intro-outer-div'),
                html.Div([modal], className='middle-toggles'),
                dcc.Store(id='store-conversation', data=''),
                dcc.Loading(id='loading_0', type='circle',
                            children=[conversation]),
                html.Div([controls], style={'margin-bottom': '20px'}),
                html.Div([
                    html.Div([badges], className='badges'),
                ], className='badges-div'),
            ], width=12)
        ], justify='center'),
    ],
)


@app.callback(
    Output('display-conversation', 'children'),
    [Input('store-conversation', 'data')]

)
def update_display(chat_history):
    return [
        textbox(chat_history, box='self') if i % 2 == 0 else textbox(
            chat_history, box='other')
        for i, chat_history in enumerate(chat_history)
    ]


@app.callback(
    [
        Output('store-conversation', 'data'),
        Output('user-input', 'value'),
        Output('display-conversation', 'value')
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
        chat_history.insert(0, f'{avatar}    👋')
        chat_history.insert(0, "🤖    Hello, how are you? My name is `Ai.ra`! I am a `language model`. More specifically, I am a machine learning model trained for conversation and Q&A (_a chatbot_). I was trained to answer questions about AI Ethics and AI Safety! Would you like a summary of the terms I am aware of?")
        return chat_history, '', ''

    if user_input is None or user_input == '':
        chat_history.insert(0, f'{avatar}    👋')
        chat_history.insert(0, "🤖    Hello, how are you? My name is `Ai.ra`! I am a `language model`. More specifically, I am a machine learning model trained for conversation and Q&A (_a chatbot_). I was trained to answer questions about AI Ethics and AI Safety! Would you like a summary of the terms I am aware of?")
        return chat_history, '', ''

    if user_input is None or user_input.isspace() is True:
        chat_history.insert(0, f'{avatar}    👋')
        chat_history.insert(0, "🤖    Hello, how are you? My name is `Ai.ra`! I am a `language model`. More specifically, I am a machine learning model trained for conversation and Q&A (_a chatbot_). I was trained to answer questions about AI Ethics and AI Safety! Would you like a summary of the terms I am aware of?")
        return chat_history, '', ''

    else:

        text = user_input.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        text = unidecode.unidecode(text)

        input_vector = bow_vectorizer.transform([text])

        output = classifier.predict(input_vector)[0]

        proba = classifier.predict_proba(input_vector)[0]

        bot_input_ids = f'''{answers[int(output)]}
        [Confidence: {max(proba) * 100: .2f} %]'''

        chat_history.insert(0, f'''{avatar}    {user_input}''')
        chat_history.insert(0, f'🤖    {bot_input_ids}')

        return chat_history, '', ''


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
