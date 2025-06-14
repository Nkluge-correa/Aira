import time
import torch
import joblib
import gradio as gr
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

dataset = load_dataset("nicholasKluge/instruct-aira-dataset", split='portuguese')

df = dataset.to_pandas()

df.columns = ['Prompt', 'Completion']

df['Cosine Similarity'] = None

prompt_tfidf_vectorizer = joblib.load('prompt_vectorizer.pkl')
prompt_tfidf_matrix = joblib.load('prompt_tfidf_matrix.pkl')

completion_tfidf_vectorizer = joblib.load('completion_vectorizer.pkl')
completion_tfidf_matrix = joblib.load('completion_tfidf_matrix.pkl')

model_id = "nicholasKluge/Aira-2-portuguese-124M"
rewardmodel_id = "nicholasKluge/RewardModelPT"
guardrail_id = "nicholasKluge/ToxiGuardrailPT"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = AutoModelForCausalLM.from_pretrained(model_id)
rewardModel = AutoModelForSequenceClassification.from_pretrained(rewardmodel_id)
guardrail = AutoModelForSequenceClassification.from_pretrained(guardrail_id)

model.eval()
rewardModel.eval()
guardrail.eval()

model.to(device)
rewardModel.to(device)
guardrail.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)
rewardTokenizer = AutoTokenizer.from_pretrained(rewardmodel_id)
guardrailTokenizer = AutoTokenizer.from_pretrained(guardrail_id)


intro = """
## O que é Aira?

[Aira](https://huggingface.co/nicholasKluge/Aira-2-portuguese-124M) é uma série de chatbots de domínio aberto (português e inglês) obtidos por meio de ajuste fino supervisionado e DPO. Aira-2 é a segunda versão da série Aira. A série Aira foi desenvolvida para ajudar os pesquisadores a explorar os desafios relacionados ao problema de alinhamento.

## Limitações

Desenvolvemos os nossos chatbots através de ajuste fino supervisionado e DPO. Esta abordagem tem muitas limitações. Apesar de podermos criar um chatbot capaz de responder a perguntas sobre qualquer assunto, é difícil forçar o modelo a produzir respostas de boa qualidade. E por boa, queremos dizer texto **factual** e **inofensivo**. Isto leva-nos a alguns problemas:

**Alucinações:** Esse modelo pode produzir conteúdo que pode ser confundido com a verdade, mas que é, de fato, enganoso ou totalmente falso, ou seja, alucinação.

**Vieses e toxicidade:** Esse modelo herda os estereótipos sociais e históricos dos dados usados para treiná-lo. Devido a esses vieses, o modelo pode produzir conteúdo tóxico, ou seja, nocivo, ofensivo ou prejudicial a indivíduos, grupos ou comunidades.

**Repetição e verbosidade:** O modelo pode ficar preso em loops de repetição (especialmente se a penalidade de repetição durante as gerações for definida com um valor escasso) ou produzir respostas prolixas sem relação com o prompt que recebeu.

## Uso Intendido

Aira destina-se apenas à investigação acadêmica. Para mais informações, leia nossa [carta modelo](https://huggingface.co/nicholasKluge/Aira-2-portuguese-124M).

## Como essa demo funciona?

Para esta demonstração, utilizamos o modelo mais leve que treinamos (Aira-2-portuguese-124M) e uma estratégia de amostragem best-of-n. Esta demonstração utiliza um [modelo de recompensa](https://huggingface.co/nicholasKluge/RewardModelPT) e uma [guardrail](https://huggingface.co/nicholasKluge/ToxiGuardrailPT) para avaliar a pontuação de cada resposta candidata, considerando o seu alinhamento com a mensagem do utilizador e o seu nível de nocividade. A função de geração organiza as respostas candidatas por ordem da sua pontuação de recompensa e elimina as respostas consideradas tóxicas ou nocivas. Posteriormente, a função de geração devolve a resposta candidata com a pontuação mais elevada que ultrapassa o limiar de segurança, ou uma mensagem pré-estabelecida se não forem identificados candidatos seguros.
"""

search_intro ="""
<h2><center>Explore o conjunto de dados de alinhamento 🔍</h2></center>

Aqui, os usuários podem procurar instâncias no conjunto de dados de ajuste fino (SFT). Para permitir uma pesquisa rápida, usamos a representação Term Frequency-Inverse Document Frequency (TF-IDF) e a similaridade de cosseno para explorar o conjunto de dados. Os vetorizadores TF-IDF pré-treinados e as matrizes TF-IDF correspondentes estão disponíveis neste repositório. Abaixo, apresentamos as dez instâncias mais semelhantes no conjunto de dados de ajuste fino utilizado. 

Os usuários podem usar essa ferramenta para explorar como o modelo interpola os dados de ajuste fino e se ele é capaz de seguir instruções que estão fora da distribuição de ajuste fino.
"""

disclaimer = """
**Isenção de responsabilidade:** Esta demonstração deve ser utilizada apenas para fins de investigação. Os moderadores não censuram a saída do modelo, e os autores não endossam as opiniões geradas por este modelo.

Se desejar apresentar uma reclamação sobre qualquer mensagem produzida pelo modelo, por favor contatar [nicholas@airespucrs.org](mailto:nicholas@airespucrs.org).
"""

with gr.Blocks(theme='freddyaboulton/dracula_revamped') as demo:

    gr.Markdown("""<h1><center>Aira Demo (Português) 🤓💬</h1></center>""")
    gr.Markdown(intro)

    
    chatbot = gr.Chatbot(label="Aira", 
                        height=500,
                        show_copy_button=True,
                        avatar_images=("./astronaut.png", "./robot.png"),
                        render_markdown= True,
                        line_breaks=True,
                        layout='panel')
                         
    msg = gr.Textbox(label="Escreva uma pergunta ou instrução para Aira ...", placeholder="Olá Aira, como vai você?")

    # Parameters to control the generation
    with gr.Accordion(label="Parâmetros ⚙️", open=False):
        safety = gr.Radio(["On", "Off"], label="Proteção 🛡️", value="On", info="Ajuda a prevenir o modelo de gerar conteúdo tóxico.")
        top_k = gr.Slider(minimum=10, maximum=100, value=50, step=5, interactive=True, label="Top-k", info="Controla o número de tokens de maior probabilidade a considerar em cada passo.")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, interactive=True, label="Top-p", info="Controla a probabilidade cumulativa dos tokens gerados.")
        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.1, step=0.1, interactive=True, label="Temperatura", info="Controla a aleatoriedade dos tokens gerados.")
        repetition_penalty = gr.Slider(minimum=1, maximum=2, value=1.1, step=0.1, interactive=True, label="Penalidade de Repetição", info="Valores mais altos auxiliam o modelo a evitar repetições na geração de texto.")
        max_new_tokens = gr.Slider(minimum=10, maximum=500, value=200, step=10, interactive=True, label="Comprimento Máximo", info="Controla o número máximo de tokens a serem produzidos (ignorando o prompt).")
        smaple_from = gr.Slider(minimum=2, maximum=10, value=2, step=1, interactive=True, label="Amostragem por Rejeição", info="Controla o número de gerações a partir das quais o modelo de recompensa irá selecionar.")

    
    clear = gr.Button("Limpar Conversa 🧹")

    gr.Markdown(search_intro)
    
    search_input = gr.Textbox(label="Cole aqui o prompt ou a conclusão que você gostaria de pesquisar...", placeholder="Qual a Capital do Brasil?")
    search_field = gr.Radio(['Prompt', 'Completion'], label="Coluna do Dataset", value='Prompt')
    submit = gr.Button(value="Buscar")

    with gr.Row():
        out_dataframe = gr.Dataframe(
            headers=df.columns.tolist(),
            datatype=["str", "str", "str"],
            row_count=10,
            col_count=(3, "fixed"),
            wrap=True,
            interactive=False
        )
    
    gr.Markdown(disclaimer)

    def user(user_message, chat_history):
        """
        Chatbot's user message handler.
        """
        return gr.update(value=user_message, interactive=True), chat_history + [[user_message, None]]

    def generate_response(user_msg, top_p, temperature, top_k, max_new_tokens, smaple_from, repetition_penalty, safety, chat_history):
        """
        Chatbot's response generator.
        """

        inputs = tokenizer(tokenizer.bos_token + user_msg + tokenizer.sep_token, 
            add_special_tokens=False,
            return_tensors="pt").to(model.device)

        generated_response = model.generate(**inputs,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            early_stopping=True,
            renormalize_logits=True,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=smaple_from)

        decoded_text = [tokenizer.decode(tokens, skip_special_tokens=True).replace(user_msg, "") for tokens in generated_response]

        rewards = []

        if safety == "On":
            guardrail_scores = []

        for text in decoded_text:
            reward_tokens = rewardTokenizer(user_msg, text,
                        truncation=True,
                        max_length=512,
                        return_token_type_ids=False,
                        return_tensors="pt",
                        return_attention_mask=True)
            
            reward_tokens.to(rewardModel.device)
            
            reward = rewardModel(**reward_tokens)[0].item()
            rewards.append(reward)

            if safety == "On":
                safety_tokens = guardrailTokenizer(user_msg + " " + text,
                            truncation=True,
                            max_length=512,
                            return_token_type_ids=False,
                            return_tensors="pt",
                            return_attention_mask=True)
                
                safety_tokens.to(guardrail.device)
                
                guardrail_score = guardrail(**safety_tokens)[0].item()
                guardrail_scores.append(guardrail_score)
                
                guardrail_threshold = 5 # Adjust this threshold based on your safety requirements

        if safety == "On":
            ordered_generations = sorted(zip(decoded_text, rewards, guardrail_scores), key=lambda x: x[1], reverse=True)
            ordered_generations = [(x, y, z) for (x, y, z) in ordered_generations if z >= guardrail_threshold]

        else:
            ordered_generations = sorted(zip(decoded_text, rewards), key=lambda x: x[1], reverse=True)

        if len(ordered_generations) == 0:
          bot_message = """Peço desculpa pelo incómodo, mas parece que não foi possível identificar respostas adequadas que cumpram as nossas normas de segurança. Infelizmente, isto indica que o conteúdo gerado pode conter elementos de toxicidade ou pode não ajudar a responder à sua mensagem. A sua opinião é valiosa para nós e esforçamo-nos por garantir uma conversa segura e construtiva. Não hesite em fornecer mais pormenores ou colocar quaisquer outras questões, e farei o meu melhor para o ajudar."""

        else:
          bot_message = ordered_generations[0][0]

        chat_history[-1][1] = ""
        for character in bot_message:
            chat_history[-1][1] += character
            time.sleep(0.005)
            yield chat_history

    def search_in_datset(column_name, search_string):
        """
        Search in the dataset for the most similar instances.
        """
        temp_df = df.copy()

        if column_name == 'Prompt':
            search_vector = prompt_tfidf_vectorizer.transform([search_string])
            cosine_similarities = cosine_similarity(prompt_tfidf_matrix, search_vector)
            temp_df['Cosine Similarity'] = cosine_similarities
            temp_df.sort_values('Cosine Similarity', ascending=False, inplace=True)
            return temp_df.head(10)
        
        elif column_name == 'Completion':
            search_vector = completion_tfidf_vectorizer.transform([search_string])
            cosine_similarities = cosine_similarity(completion_tfidf_matrix, search_vector)
            temp_df['Cosine Similarity'] = cosine_similarities
            temp_df.sort_values('Cosine Similarity', ascending=False, inplace=True)
            return temp_df.head(10)

    
    
    response = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        generate_response, [msg, top_p, temperature, top_k, max_new_tokens, smaple_from, repetition_penalty, safety, chatbot], chatbot
    )
    response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
    msg.submit(lambda x: gr.update(value=''), None,[msg])
    clear.click(lambda: None, None, chatbot, queue=False)
    submit.click(fn=search_in_datset, inputs=[search_field, search_input], outputs=out_dataframe)

demo.queue()
demo.launch()
