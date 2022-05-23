import tensorflow as tf
# noinspection PyUnresolvedReferences
import tensorflow.keras as keras
# noinspection PyUnresolvedReferences
import tensorflow.keras.layers as layers
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
# Regular variable assignments not working???
# noinspection PyStatementEffect
(df := pd.read_csv("data/news.csv"))
df.drop(columns=["Unnamed: 0"], inplace=True)
X = df.drop(columns=["label"])
y = df.drop(columns=["title", "text"])
# %%
y["label"] = y["label"].apply(lambda x: 0 if x == "FAKE" else 1)
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# %%
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.string)
# %%
# title,text,label
title_vectorizer = layers.TextVectorization(
    output_sequence_length=100,
    max_tokens=1000,
    pad_to_max_tokens=True,
)
text_vectorizer = layers.TextVectorization(
    output_sequence_length=100,
    max_tokens=1000,
    pad_to_max_tokens=True,
)
# %%
title_vectorizer.adapt(X_train_tensor[0])
text_vectorizer.adapt(X_train_tensor[1])
# %%
inp = layers.Input(shape=(2,), dtype=tf.string)
title = layers.Lambda(lambda x: x[::1, :1])(inp)
text = layers.Lambda(lambda x: x[::1, 1:])(inp)

title_vec = title_vectorizer(title)
text_vec = text_vectorizer(text)

out = layers.concatenate([title_vec, text_vec])
out = layers.Dense(32, activation="relu")(out)
out = layers.Dense(64)(out)
out = layers.Dense(1, activation="sigmoid")(out)
model = keras.Model(inputs=inp, outputs=out)
# %%
model.summary()
# %%
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0),
)
# %%
history = model.fit(
    tf.convert_to_tensor(X_train),
    tf.convert_to_tensor(y_train),
    epochs=10,
    batch_size=32,
)
# %%
results = model.evaluate(
    tf.convert_to_tensor(X_test),
    tf.convert_to_tensor(y_test),
    batch_size=32,
)
print("test loss, test acc:", results)
#%%
model.save("models/fake_news_v1")

