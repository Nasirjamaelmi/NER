import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset

# Load Swedish NER model
model_name = "KBLab/bert-base-swedish-cased-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create the NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
ds = load_dataset("rjac/kaggle-entity-annotated-corpus-ner-dataset")
# Load CSV
df = pd.read_csv("train.csv")

# Process each row and extract entities
results = []
for _, row in df.iterrows():
    text = row["text"]
    entities = ner_pipeline(text)
    entity_list = [f"{e['word']} ({e['entity_group']})" for e in entities]
    results.append(", ".join(entity_list))

# Add entities to a new column
df["entities"] = results

# Save or print the result
df.to_csv("ner_output.csv", index=False)
print(df)

