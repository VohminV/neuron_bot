import json
import os
import wikipedia
import random
from collections import defaultdict
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters
)

# 🔧 Настройки
BOT_TOKEN = "ТВОЙ_ТОКЕН_ЗДЕСЬ"  # ← сюда вставь токен от BotFather
BRAIN_FILE = "brain.json"
SYN_FILE = "synonyms.json"

wikipedia.set_lang("ru")

if os.path.exists(SYN_FILE):
    with open(SYN_FILE, "r", encoding="utf-8") as f:
        SYNONYMS = json.load(f)
else:
    SYNONYMS = {
        "привет": ["здравствуй", "хай", "добрый день", "здорово"],
        "как дела": ["как ты", "как жизнь", "что нового"],
        "пока": ["до свидания", "бай", "чао", "увидимся"]
    }

class MarkovChain:
    def __init__(self):
        self.model = defaultdict(list)

    def train(self, text):
        words = text.strip().split()
        for i in range(len(words) - 1):
            self.model[words[i]].append(words[i + 1])

    def generate(self, length=10):
        if not self.model:
            return ""
        word = random.choice(list(self.model.keys()))
        result = [word]
        for _ in range(length - 1):
            next_words = self.model.get(word)
            if not next_words:
                break
            word = random.choice(next_words)
            result.append(word)
        return " ".join(result)

class Neuron:
    def __init__(self, phrase):
        self.phrase = phrase
        self.connections = {}

    def connect(self, response_phrase, weight=1):
        if response_phrase in self.connections:
            self.connections[response_phrase] += weight
        else:
            self.connections[response_phrase] = weight

    def get_best_response(self):
        if not self.connections:
            return None
        return max(self.connections, key=self.connections.get)

    def generate_response(self):
        if not self.connections:
            return None
        responses = list(self.connections.keys())
        weights = list(self.connections.values())
        selected = random.choices(responses, weights=weights, k=1)[0]
        if len(responses) > 1:
            other = random.choice(responses)
            if other != selected:
                return f"{selected}. А ещё: {other}"
        return selected

class Brain:
    def __init__(self):
        self.neurons = {}
        self.markov = MarkovChain()

    def get_or_create(self, phrase):
        phrase = phrase.lower()
        if phrase not in self.neurons:
            self.neurons[phrase] = Neuron(phrase)
        return self.neurons[phrase]

    def get_all_synonyms(self, phrase):
        phrase = phrase.lower()
        variants = {phrase}
        for key, values in SYNONYMS.items():
            if phrase == key or phrase in values:
                variants.update([key] + values)
        return list(variants)

    def respond(self, phrase):
        for variant in self.get_all_synonyms(phrase):
            neuron = self.neurons.get(variant)
            if neuron:
                response = neuron.get_best_response()
                if response:
                    return response
        return None

    def train(self, input_phrase, response_phrase):
        for variant in self.get_all_synonyms(input_phrase):
            neuron = self.get_or_create(variant)
            neuron.connect(response_phrase)
        self.markov.train(response_phrase)

    def save(self):
        data = {p: n.connections for p, n in self.neurons.items()}
        with open(BRAIN_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if not os.path.exists(BRAIN_FILE):
            return
        with open(BRAIN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            for phrase, connections in data.items():
                neuron = self.get_or_create(phrase)
                neuron.connections = connections

    def web_learn(self, phrase):
        try:
            summary = wikipedia.summary(phrase, sentences=2)
            self.train(phrase, summary)
            return summary
        except:
            return "Не удалось найти ответ в сети."

    def generate_response(self, phrase):
        for variant in self.get_all_synonyms(phrase):
            neuron = self.neurons.get(variant)
            if neuron:
                response = neuron.generate_response()
                if response:
                    return response
        return None

    def generate_markov(self):
        return self.markov.generate()

brain = Brain()
brain.load()
learning_users = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот с нейронами и цепями Маркова. Напиши что-нибудь!")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip().lower()

    if user_id in learning_users:
        original_phrase = learning_users[user_id]
        brain.train(original_phrase, text)
        brain.save()
        del learning_users[user_id]
        await update.message.reply_text("Запомнил! Спасибо.")
        return

    response = brain.generate_response(text)
    if not response:
        response = brain.generate_markov()

    if response:
        await update.message.reply_text(response)
    else:
        await update.message.reply_text("Я не знаю, как ответить. Как бы ты ответил?")
        learning_users[user_id] = text

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    print("🤖 Бот запущен!")
    app.run_polling()

if __name__ == "__main__":
    main()