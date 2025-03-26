import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# تهيئة Lemmatizer
lemmatizer = WordNetLemmatizer()

# تحميل بيانات النوايا
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

# تدريب النموذج (إذا لم يكن مدربًا مسبقًا)
def train_model():
    words = []
    labels = []
    docs = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            docs.append((tokens, intent["tag"]))
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in "؟?!"]
    words = sorted(set(words))
    labels = sorted(set(labels))

    training = []
    output_empty = [0] * len(labels)

    for doc in docs:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
        for w in words:
            bag.append(1) if w in word_patterns else bag.append(0)
        output_row = list(output_empty)
        output_row[labels.index(doc[1])] = 1
        training.append([bag, output_row])

    train_x = np.array([i[0] for i in training])
    train_y = np.array([i[1] for i in training])

    # بناء النموذج
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    model.save("chatbot_model.h5")
    pickle.dump((words, labels, train_x, train_y), open("chatbot_data.pkl", "wb"))

# تحميل النموذج والبيانات (إذا كان مدربًا مسبقًا)
def load_model():
    model = keras.models.load_model("chatbot_model.h5")
    words, labels, _, _ = pickle.load(open("chatbot_data.pkl", "rb"))
    return model, words, labels

# إعداد FastAPI
app = FastAPI()

# نموذج لطلب المستخدم
class UserMessage(BaseModel):
    message: str

# دالة لتحويل النص إلى حقائب كلمات
def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = word_tokenize(s)
    s_words = [lemmatizer.lemmatize(w.lower()) for w in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# دالة للحصول على الرد
def get_response(text, model, words, labels):
    X_input = np.array([bag_of_words(text, words)])
    predictions = model.predict(X_input)[0]
    index = np.argmax(predictions)
    tag = labels[index]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "أنا غير متأكد من كيفية الرد على ذلك."

# نقطة النهاية لـ API
@app.post("/chat/")
async def chat(user_message: UserMessage):
    model, words, labels = load_model()
    user_text = user_message.message
    response = get_response(user_text, model, words, labels)
    return {"response": response}

# تشغيل التدريب أو الـ API
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
