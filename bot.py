import os
import re
import wikipedia
import torch
import torch.nn.functional as F
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
BOT_TOKEN = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wikipedia.set_lang("ru")  # Русский язык

# ======== Функции для Википедии ========
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
    reasons = [s for s in sentences if any(p in s.lower() for p in patterns)]
    if reasons:
        return " ".join(reasons)
    return None

def extract_who(text):
    sentences = re.split(r"(?<=[.!?]) +", text)
    for s in sentences:
        if re.search(r"[А-ЯЁ][а-яё]+", s):
            return s
    return None

def extract_when(text):
    sentences = re.split(r"(?<=[.!?]) +", text)
    date_patterns = [r"\b\d{4}\b", r"\b\d{1,2} [а-я]+ \d{4}\b"]
    for s in sentences:
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
            else:
                sentences = re.split(r"(?<=[.!?]) +", summary)
                return "Из Википедии:\n" + " ".join(sentences[:3])
        return "Из Википедии:\n" + wikipedia.summary(results[0], sentences=3)
    except Exception:
        return None

# ======== Класс с генерацией и обучением ========
class OnlineTrainer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if os.path.exists(FINETUNED_MODEL_DIR):
            self.model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_DIR)
            print("Загружена дообученная модель")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            print("Загружена базовая модель distilgpt2")

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(device)

        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def generate_answer(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
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

        # Получаем вероятности токенов для сгенерированного текста
        probs = self.get_token_probabilities(generated_text)

        return generated_text, probs

    def get_token_probabilities(self, text):
        """
        Прогоняем текст через модель с labels=input_ids, чтобы получить логи вероятностей токенов.
        Возвращаем список вероятностей для каждого токена.
        """
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        probs = torch.softmax(logits, dim=-1)

        token_probs = []
        for i in range(input_ids.size(1) - 1):
            token_id = input_ids[0, i + 1].item()
            prob = probs[0, i, token_id].item()
            token_probs.append(prob)

        return token_probs

    def fine_tune(self, question, answer, epochs=1):
        text = f"User: {question}\nBot: {answer}\n"
        encodings = self.tokenizer(text, return_tensors="pt", padding=True)
        dataset = Dataset.from_dict(encodings)

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

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self.data_collator,
        )

        trainer.train()
        trainer.save_model(FINETUNED_MODEL_DIR)
        print("Модель дообучена и сохранена")

trainer = OnlineTrainer()
learning_users = {}
trained_questions = set()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я ИИ-бот с поиском по Википедии и онлайн обучением.")

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

    if token_probs:
        avg_prob = sum(token_probs) / len(token_probs)
    else:
        avg_prob = 0.0  # на всякий случай

    PROB_THRESHOLD = 0.5  # порог уверенности (средняя вероятность токена)

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
