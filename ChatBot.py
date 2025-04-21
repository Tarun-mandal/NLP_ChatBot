# === 1) Setup & Imports ===
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import openai
import torch

# === 2) Load PersonaChat & Sample 2k Examples ===
dataset = load_dataset("bavard/personachat_truecased")

max_train_samples = 1800
max_val_samples   = 200

def format_example(ex):
    history = ex["history"]
    input_text  = "\n".join(history)
    target_text = ex["candidates"][0]
    return {"input_text": input_text, "target_text": target_text}

train_raw = dataset["train"].select(range(max_train_samples)).map(format_example)
val_raw   = dataset["validation"].select(range(max_val_samples)).map(format_example)

# === 3) Initialize BERT2BERT Summarizer ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "bert-base-uncased",
    "bert-base-uncased"
)

# tie embeddings and set special tokens
model.config.tie_encoder_decoder      = True
model.config.is_encoder_decoder       = True
model.config.decoder_start_token_id   = tokenizer.cls_token_id
model.config.bos_token_id             = tokenizer.cls_token_id
model.config.eos_token_id             = tokenizer.sep_token_id
model.config.pad_token_id             = tokenizer.pad_token_id

# === 4) Tokenization ===
max_input_length  = 512
max_target_length = 64

def tokenize_fn(ex):
    enc = tokenizer(
        ex["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )
    dec = tokenizer(
        ex["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )
    enc["labels"] = dec["input_ids"]
    return enc

train_ds = train_raw.map(tokenize_fn, remove_columns=train_raw.column_names)
val_ds   = val_raw.map(tokenize_fn,   remove_columns=val_raw.column_names)

# === 5) Trainer & TrainingArguments ===
training_args = TrainingArguments(
    output_dir="./bert2bert_personachat",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# === 6) Fine‑tune Summarizer ===
trainer.train()
model.save_pretrained("bert2bert_personachat_finetuned")
tokenizer.save_pretrained("bert2bert_personachat_finetuned")

# === 7) Load Summarizer & Set OpenAI Key ===
openai.api_key = "sk-..."  # ← replace with your key

summ_tokenizer = BertTokenizer.from_pretrained("bert2bert_personachat_finetuned")
summ_model     = EncoderDecoderModel.from_pretrained("bert2bert_personachat_finetuned")

# ** Ensure special tokens are set on loaded model **
summ_model.config.is_encoder_decoder       = True
summ_model.config.decoder_start_token_id   = summ_tokenizer.cls_token_id
summ_model.config.bos_token_id             = summ_tokenizer.cls_token_id
summ_model.config.eos_token_id             = summ_tokenizer.sep_token_id
summ_model.config.pad_token_id             = summ_tokenizer.pad_token_id

# (Optional) verify
print("decoder_start_token_id:", summ_model.config.decoder_start_token_id)
print("bos_token_id:          ", summ_model.config.bos_token_id)
print("eos_token_id:          ", summ_model.config.eos_token_id)
print("pad_token_id:          ", summ_model.config.pad_token_id)

# === 8) Summarization Function ===
def summarize_conversation_bert(history):
    text = "\n".join(history)
    inputs = summ_tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    summary_ids = summ_model.generate(
        inputs["input_ids"],
        decoder_start_token_id=summ_model.config.decoder_start_token_id,
        bos_token_id=summ_model.config.bos_token_id,
        eos_token_id=summ_model.config.eos_token_id,
        pad_token_id=summ_model.config.pad_token_id,
        max_length=64,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# === 9) ChatGPT‑3.5 Response Generator ===
# === 9) ChatGPT‑3.5 Response Generator (v1.0+ API) ===
def generate_response_gpt35(summary, user_input):
    prompt = (
        f"Conversation summary:\n{summary}\n\n"
        f"User just said: \"{user_input}\"\n"
        "Bot reply:"
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=50,
    )
    return resp.choices[0].message.content.strip()


# === 10) Run the Prompt‑Based Chatbot ===
def run_chatbot():
    print("Bot: Hello! (type 'exit' to quit)")
    history = ["Bot: Hello!"]
    while True:
        user_input = input("User: ")
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Bot: Goodbye!")
            break

        history.append(f"User: {user_input}")

        # summarize with our fine‑tuned BERT2BERT
        summary  = summarize_conversation_bert(history)
        # generate next turn via GPT‑3.5
        bot_reply = generate_response_gpt35(summary, user_input)

        print(f"Bot: {bot_reply}")
        history.append(f"Bot: {bot_reply}")

if __name__ == "__main__":
    run_chatbot()