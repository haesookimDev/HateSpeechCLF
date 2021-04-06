import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv("./hate_speech_binary_dataset.csv")
data["문장"] = data["문장"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
data["문장"] = data["문장"].replace("", np.nan)
data.dropna(inplace=True)
data.drop_duplicates(subset=["문장"], inplace=True)

print(len(data))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data["문장"], data["혐오 여부"],
                                                    test_size=0.2, random_state=100)


def jamo_split(s:str) -> str:
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    JONGSUNG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    s_code = ord(s)

    if '가' <= s <= '힣':
        return CHOSUNG_LIST[(s_code - 0xAC00) // (28 * 21)] + JUNGSUNG_LIST[((s_code - 0xAC00) // 28) % 21]                + JONGSUNG_LIST[(s_code - 0xAC00) % 28]
    else:
        return s


def string_jamo_split(s:str) -> str:
    result = ""
    for i in list(s.strip()):
        result += jamo_split(i)

    return result


print(string_jamo_split("안녕하세요"))

X_train_jamo_split = []
for i in X_train:
    X_train_jamo_split.append(string_jamo_split(i))

print(X_train_jamo_split[:15])

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train_jamo_split)

encoded = np.array(tokenizer.texts_to_sequences(X_train_jamo_split))

plt.hist([len(x) for x in encoded])
plt.show()

encoded = pad_sequences(encoded, maxlen=500)

from tensorflow import keras
model = keras.models.Sequential()
model.add(keras.layers.Embedding(500, 100))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(80, return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(80)))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))

es = keras.callbacks.EarlyStopping(monitor="loss", mode="min", verbose=1, patience=4)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])

model.fit(encoded, y_train, epochs=15, callbacks=[es], validation_split=0.2)

def predict(new_sentence):
    # 0에 가까울수록 악성 댓글임
    new_sentence = string_jamo_split(new_sentence)
    encoded = tokenizer.texts_to_sequences([new_sentence])
    pad_new = pad_sequences(encoded, maxlen = 500)
    return float(model.predict(pad_new))

model.save("./model.h5")


