import os
import re
import wikipedia
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

MODEL_NAME = "distilgpt2"
FINETUNED_MODEL_DIR = "./finetuned_model"
BOT_TOKEN = ""  # вставь свой токен

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wikipedia.set_lang("ru")

# === Функции для Википедии ===
def detect_question_type(text):
    text = text.lower()
    if any(w in text for w in ["почему", "зачем", "из-за", "потому что"]):
        return "why"
    if any(w in text for w in ["кто", "кем", "чей"]):
        return "who"
    if any(w in text for w in ["когда", "дата", "год", "число"]):
        return "when"
    if any(w in text for w in ["где", "куда", "откуда"]):
        return "where"
    if any(w in text for w in ["как", "каким образом"]):
        return "how"
    return "what"

def extract_reason(text):
    patterns = ["потому что", "из-за", "так как", "вследствие", "причина", "по причине"]
    sentences = re.split(r"(?<=[.!?]) +", text)
    return " ".join(s for s in sentences if any(p in s.lower() for p in patterns)) or None

def extract_who(text):
    for s in re.split(r"(?<=[.!?]) +", text):
        if re.search(r"[А-ЯЁ][а-яё]+", s):
            return s
    return None

def extract_when(text):
    date_patterns = [r"\b\d{4}\b", r"\b\d{1,2} [а-я]+ \d{4}\b"]
    for s in re.split(r"(?<=[.!?]) +", text):
        if any(re.search(p, s) for p in date_patterns):
            return s
    return None

def search_wikipedia(phrase):
    try:
        results = wikipedia.search(phrase)
        if not results:
            return None
        qtype = detect_question_type(phrase)
        for title in results:
            summary = wikipedia.summary(title, sentences=6)
            if qtype == "why":
                reason = extract_reason(summary)
                if reason:
                    return f"Из Википедии:\n{reason}"
            elif qtype == "who":
                who = extract_who(summary)
                if who:
                    return f"Из Википедии:\n{who}"
            elif qtype == "when":
                when = extract_when(summary)
                if when:
                    return f"Из Википедии:\n{when}"
            return "Из Википедии:\n" + " ".join(re.split(r"(?<=[.!?]) +", summary)[:3])
        return "Из Википедии:\n" + wikipedia.summary(results[0], sentences=3)
    except Exception:
        return None

# === Класс OnlineTrainer с chunk-файнтюнингом ===
class OnlineTrainer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL_DIR if os.path.exists(FINETUNED_MODEL_DIR) else MODEL_NAME
        )
        print("Загружена дообученная модель" if os.path.exists(FINETUNED_MODEL_DIR) else "Загружена базовая модель")

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(device)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def chunk_text(self, text, max_length=1024):
        tokens = self.tokenizer.encode(text, truncation=False)
        return [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

    def fine_tune(self, question, answer, epochs=1):
        full_text = f"User: {question}\nBot: {answer}\n"
        chunks = self.chunk_text(full_text, max_length=1024)

        training_args = TrainingArguments(
            output_dir=FINETUNED_MODEL_DIR,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            save_steps=10,
            save_total_limit=2,
            logging_steps=5,
            logging_dir='./logs',
            no_cuda=(device.type != "cuda"),
        )

        for i, chunk in enumerate(chunks):
            dataset = Dataset.from_dict({"input_ids": [chunk]})
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=self.data_collator,
            )
            print(f"Дообучение на чанке {i + 1}/{len(chunks)}...")
            trainer.train()

        self.model.save_pretrained(FINETUNED_MODEL_DIR)
        print("✅ Модель дообучена и сохранена.")

    def generate_answer(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.7,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = answer[len(prompt):].strip()
        probs = self.get_token_probabilities(generated_text)
        return generated_text, probs

    def get_token_probabilities(self, text):
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        token_probs = []
        for i in range(input_ids.size(1) - 1):
            token_id = input_ids[0, i + 1].item()
            prob = probs[0, i, token_id].item()
            token_probs.append(prob)
        return token_probs

# === Telegram боты ===
trainer = OnlineTrainer()
learning_users = {}
trained_questions = set()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я ИИ-бот с Википедией и онлайн обучением.")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip()

    if user_id in learning_users:
        question = learning_users[user_id]
        answer = text
        await update.message.reply_text("Спасибо! Я запомнил ответ и сейчас обучусь.")
        trainer.fine_tune(question, answer)
        await update.message.reply_text("Обучение завершено.")
        del learning_users[user_id]
        return

    wiki_answer = search_wikipedia(text)
    prompt = f"User: {text}\n"
    model_answer, token_probs = trainer.generate_answer(prompt)

    avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0.0
    PROB_THRESHOLD = 0.5

    if avg_prob < PROB_THRESHOLD and wiki_answer:
        await update.message.reply_text(wiki_answer)
        if text not in trained_questions:
            trainer.fine_tune(text, wiki_answer)
            trained_questions.add(text)
        return

    if model_answer and len(model_answer) > 15:
        await update.message.reply_text(model_answer)
        if text not in trained_questions:
            trainer.fine_tune(text, model_answer)
            trained_questions.add(text)
        return

    await update.message.reply_text("Я пока не знаю, как ответить. Как бы ты ответил?")
    learning_users[user_id] = text

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    print("🤖 Бот с Википедией и обучением запущен!")
    app.run_polling()

if __name__ == "__main__":
    main()
