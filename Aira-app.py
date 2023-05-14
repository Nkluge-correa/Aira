from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
import dash
import json

from utilities import rule_based_prediction_app, toggle_modal

with open('data/original_data/answers_en.txt', encoding='utf-8') as fp:
    answers = [line.strip() for line in fp]
    fp.close()

with open('data/generated_data/keys_en.json', 'rb') as fp:
    vocabulary = json.load(fp)
    fp.close()


def textbox(text, box="other"):
    style = {
        'max-width': '100%',
        'width': 'max-content',
        'padding': '10px 10px',
        'margin-bottom': '20px',
        'text-align': 'justify',
        'text-justify': 'inter-word',
        'font-size': '1.em',
        'backdrop-filter': 'blur(14px)'

    }

    if box == "self":
        style["margin-left"] = "auto"
        style["margin-right"] = 0
        style["border-radius"] = '10px 10px 10px 0px'
        style["box-shadow"] = '-5px 10px 8px #1c1c1b'

        color = "primary"
        inverse = True

    elif box == "other":
        style["margin-left"] = 0
        style["margin-right"] = "auto"
        style["border-radius"] = '10px 10px 0px 10px'
        style["box-shadow"] = '5px 10px 8px #1c1c1b'

        color = "secondary"
        inverse = True

    else:
        raise ValueError("Incorrect option for `box`.")

    return dbc.Card(dcc.Markdown(text), style=style, body=True, color=color, inverse=inverse)


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
        dcc.Textarea(
            id='user-input',
            value='',
            title='''Write to Aira...''',
            style={'border-radius': '5px',
                   'margin': '2px',
                   'color': '#313638',
                   'background-color': '#f2f2f2',
                   'width': '100%',
                   'max-width': '100vw', }),
        dbc.Button(
            [html.I(className="bi bi-send")], size='lg', id='submit',
            outline=True, color='light', style={
                'border-radius': '5px',
                'margin': '2px',
                'width': '100%',
                'max-width': '100vw',
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
                    '## What is `Aira`? 🤔'), style={'font-weight': 'bold'})),
                dbc.ModalBody([
                    dcc.Markdown(
                        '''`Aira` was developed by [`Nicholas Kluge`](https://nkluge-correa.github.io/) and [`Carolina Del Pino`](http://lattes.cnpq.br/6291330432531578). For more information visit this [`repository`](https://github.com/Nkluge-correa/Aira-EXPERT).''', className='modal-body-text-style', style={'font-size': '1rem'}),
                ]),
                dbc.ModalFooter(
                    dbc.Button(
                        html.I(className="bi bi-x-circle"),
                        id='close-body-scroll',
                        className='ms-auto',
                        outline=True,
                        size='xl',
                        n_clicks=0,
                        color='primary',
                        style={'border': 0}
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

badges = html.Span([
    dbc.Badge([html.I(className="bi bi-heart-fill"), "  Open-Source"], href="https://github.com/Nkluge-correa/Aira-EXPERT",
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
app.title = 'Aira Expert 🤓'


app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([
            dbc.Col([
                html.Div([dcc.Markdown('# `Aira Expert`', className='title-style'),
                          html.Img(src=dash.get_asset_url(
                              'chat.gif'), height="80px", className='title-icon-style')],
                    className='title-div'),
                html.Div([
                    html.Div([
                        dcc.Markdown('''
                Aira is a chatbot designed to simulate the way an expert would behave during a round of questions and answers.
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
    [Input('store-conversation', 'data')],

)
def update_display(chat_history):
    return [
        textbox(x, box="self") if i % 2 != 0 else textbox(x, box="other")
        for i, x in enumerate(chat_history)
    ]


@app.callback(
    [
        Output('store-conversation', 'data'),
        Output('user-input', 'value'),
        Output('display-conversation', 'key')
    ],

    [
        Input('submit', 'n_clicks'),
    ],

    [
        State('user-input', 'value'),
        State('store-conversation', 'data')
    ]
)
def run_chatbot(n_clicks, user_input, chat_history):
    chat_history = chat_history or []

    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    if user_input is None or user_input == '':
        raise dash.exceptions.PreventUpdate

    if user_input is None or user_input.isspace() is True:
        raise dash.exceptions.PreventUpdate

    else:

        bot_response = rule_based_prediction_app(
            user_input, vocabulary, answers)

        chat_history.insert(0, f'''👤    {user_input}''')
        chat_history.insert(0, f'🦾🤖    {bot_response}')
        return chat_history, '', ''


@app.callback(
    Output('modal-body-scroll', 'is_open'),
    [
        Input('open-body-scroll', 'n_clicks'),
        Input('close-body-scroll', 'n_clicks'),
    ],
    [State('modal-body-scroll', 'is_open')],
)
def toggle_aira(n1, n2, is_open):
    return toggle_modal(n1, n2, is_open)


if __name__ == '__main__':
    app.run_server(debug=False)
