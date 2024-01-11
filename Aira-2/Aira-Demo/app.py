import time
import torch
import joblib
import gradio as gr
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


# download the instruct-aira-dataset
dataset = load_dataset("nicholasKluge/instruct-aira-dataset", split='english')

# convert the dataset to a pandas dataframe
df = dataset.to_pandas()

# rename the columns
df.columns = ['Prompt', 'Completion']

# add a column to store the cosine similarity
df['Cosine Similarity'] = None

# Load the saved prompt TfidfVectorizer
prompt_tfidf_vectorizer = joblib.load('prompt_vectorizer.pkl')

# load the prompt tfidf_matrix
prompt_tfidf_matrix = joblib.load('prompt_tfidf_matrix.pkl')

# Load the saved completion TfidfVectorizer
completion_tfidf_vectorizer = joblib.load('completion_vectorizer.pkl')

# load the completion tfidf_matrix
completion_tfidf_matrix = joblib.load('completion_tfidf_matrix.pkl')

# specify the model's ids
model_id = "nicholasKluge/Aira-OPT-125M"
rewardmodel_id = "nicholasKluge/RewardModel"
toxicitymodel_id = "nicholasKluge/ToxicityModel"

# specify the device (cuda if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the models (chatbot, reward model, toxicity model)
model = AutoModelForCausalLM.from_pretrained(model_id)
rewardModel = AutoModelForSequenceClassification.from_pretrained(rewardmodel_id)
toxicityModel = AutoModelForSequenceClassification.from_pretrained(toxicitymodel_id)

# set the models to evaluation mode
model.eval()
rewardModel.eval()
toxicityModel.eval()

# set the models to the device
model.to(device)
rewardModel.to(device)
toxicityModel.to(device)

# load the tokenizers
tokenizer = AutoTokenizer.from_pretrained(model_id)
rewardTokenizer = AutoTokenizer.from_pretrained(rewardmodel_id)
toxiciyTokenizer = AutoTokenizer.from_pretrained(toxicitymodel_id)


intro = """
## What is `Aira`?

[`Aira`](https://huggingface.co/nicholasKluge/Aira-OPT-125M) is a series of open-domain chatbots (Portuguese and English) achieved via `instruction-tuning` and `RLHF`. Aira-2 is the second version of the Aira series. The Aira series was developed to help researchers explore the challenges related to the Alignment problem.

## Limitations

We developed our open-domain conversational chatbots via instruction-tuning. This approach has a lot of limitations. Even though we can make a chatbot that can answer questions about anything, forcing the model to produce good-quality responses is hard. And by good, we mean **factual** and **nontoxic**  text. This leads us to two of the most common problems with generative models used in conversational applications:

**Hallucinations:** This model can produce content that can be mistaken for truth but is, in fact, misleading or entirely false, i.e., hallucination.

**Biases and Toxicity:** This model inherits the social and historical stereotypes from the data used to train it. Given these biases, the model can produce toxic content, i.e., harmful, offensive, or detrimental to individuals, groups, or communities.

**Repetition and Verbosity:** The model may get stuck on repetition loops (especially if the repetition penalty during generations is set to a meager value) or produce verbose responses unrelated to the prompt it was given.

## Intended Use

`Aira` is intended only for academic research. For more information, read our [model card](https://huggingface.co/nicholasKluge/Aira-OPT-125M) to see how we developed `Aira`.

## How this demo works?

For this demo, we use the lighter model we have trained from the OPT series (`Aira-OPT-125M`). This demo employs a [`reward model`](https://huggingface.co/nicholasKluge/RewardModel) and a [`toxicity model`](https://huggingface.co/nicholasKluge/ToxicityModel) to evaluate the score of each candidate's response, considering its alignment with the user's message and its level of toxicity. The generation function arranges the candidate responses in order of their reward scores and eliminates any responses deemed toxic or harmful. Subsequently, the generation function returns the candidate response with the highest score that surpasses the safety threshold, or a default message if no safe candidates are identified.
"""

search_intro ="""
<h2><center>Explore Aira's Dataset 🔍</h2></center>

Here, users can look for instances in Aira's fine-tuning dataset where a given prompt or completion resembles an instruction in its dataset. To enable a fast search, we use the Term Frequency-Inverse Document Frequency (TF-IDF) representation and cosine similarity to explore the dataset. The pre-trained TF-IDF vectorizers and corresponding TF-IDF matrices are available in this repository. Below, we present the top five most similar instances in Aira's dataset for every search query.

Users can use this to explore how the model interpolates on the fine-tuning data and if it is capable of following instructions that are out of the fine-tuning distribution.
"""

disclaimer = """
**Disclaimer:** You should use this demo for research purposes only. Moderators do not censor the model output, and the authors do not endorse the opinions generated by this model.

If you would like to complain about any message produced by `Aira`, please contact [nicholas@airespucrs.org](mailto:nicholas@airespucrs.org).
"""

with gr.Blocks(theme='freddyaboulton/dracula_revamped') as demo:

    gr.Markdown("""<h1><center>Aira Demo 🤓💬</h1></center>""")
    gr.Markdown(intro)

    
    chatbot = gr.Chatbot(label="Aira", 
                        height=500,
                        show_copy_button=True,
                        avatar_images=("./astronaut.png", "./robot.png"),
                        render_markdown= True,
                        line_breaks=True,
                        likeable=False,
                        layout='panel')
    
    msg = gr.Textbox(label="Write a question or instruction to Aira ...", placeholder="What is the capital of Brazil?")

    # Parameters to control the generation
    with gr.Accordion(label="Parameters ⚙️", open=False):
        safety = gr.Radio(["On", "Off"], label="Guard Rail 🛡️", value="On", info="Helps prevent the model from generating toxic/harmful content.")
        top_k = gr.Slider(minimum=10, maximum=100, value=30, step=5, interactive=True, label="Top-k", info="Controls the number of highest probability tokens to consider for each step.")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.30, step=0.05, interactive=True, label="Top-p", info="Controls the cumulative probability of the generated tokens.")
        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.1, step=0.1, interactive=True, label="Temperature", info="Controls the randomness of the generated tokens.")
        repetition_penalty = gr.Slider(minimum=1, maximum=2, value=1.1, step=0.1, interactive=True, label="Repetition Penalty", info="Higher values help the model to avoid repetition in text generation.")
        max_new_tokens = gr.Slider(minimum=10, maximum=500, value=200, step=10, interactive=True, label="Max Length", info="Controls the maximum number of new token (not considering the prompt) to generate.")
        smaple_from = gr.Slider(minimum=2, maximum=10, value=2, step=1, interactive=True, label="Sample From", info="Controls the number of generations that the reward model will sample from.")

    clear = gr.Button("Clear Conversation 🧹")

    gr.Markdown(search_intro)
    
    search_input = gr.Textbox(label="Paste here the prompt or completion you would like to search ...", placeholder="What is the Capital of Brazil?")
    search_field = gr.Radio(['Prompt', 'Completion'], label="Dataset Column", value='Prompt')
    submit = gr.Button(value="Search")

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
            repetition_penalty=repetition_penalty,
            do_sample=True,
            early_stopping=True,
            renormalize_logits=True,
            length_penalty=0.3, 
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=smaple_from)

        decoded_text = [tokenizer.decode(tokens, skip_special_tokens=True).replace(user_msg, "") for tokens in generated_response]

        rewards = list()

        if safety == "On":
            toxicities = list()

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

                toxicity_tokens = toxiciyTokenizer(user_msg + " " + text,
                            truncation=True,
                            max_length=512,
                            return_token_type_ids=False,
                            return_tensors="pt",
                            return_attention_mask=True)
                
                toxicity_tokens.to(toxicityModel.device)
                
                toxicity = toxicityModel(**toxicity_tokens)[0].item()
                toxicities.append(toxicity)
                
                toxicity_threshold = 5

        if safety == "On":
            ordered_generations = sorted(zip(decoded_text, rewards, toxicities), key=lambda x: x[1], reverse=True)
            ordered_generations = [(x, y, z) for (x, y, z) in ordered_generations if z >= toxicity_threshold]

        else:
            ordered_generations = sorted(zip(decoded_text, rewards), key=lambda x: x[1], reverse=True)

        if len(ordered_generations) == 0:
          bot_message = """I apologize for the inconvenience, but it appears that no suitable responses meeting our safety standards could be identified. Unfortunately, this indicates that the generated content may contain elements of toxicity or may not help address your message. Your input is valuable to us, and we strive to ensure a safe and constructive conversation. Please feel free to provide further details or ask any other questions, and I will do my best to assist you."""

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