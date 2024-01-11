from datasets import DatasetDict
from datasets import Dataset
import pandas as pd

english = pd.read_parquet('aira_instruct_english.parquet')
portuguese = pd.read_parquet('aira_instruct_portuguese.parquet')
spanish = pd.read_parquet('aira_instruct_spanish.parquet')

display(portuguese)
display(english)
display(spanish)

portuguese = Dataset.from_pandas(portuguese)
english = Dataset.from_pandas(english)
spanish = Dataset.from_pandas(spanish)

ddict = DatasetDict({
    "portuguese": portuguese,  
    "english": english,
    "spanish": spanish
})

display(ddict)
ddict.push_to_hub("nicholasKluge/instruct-aira-dataset")