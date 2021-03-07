import tensorflow as tf
import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

def merge_genres(genres):
    d = dict()
    count = 0

    for genre in genres:
        count += 1
        for i in range(len(genre)):
            if i in d:
                d[i].append(genre[i])
            else:
                d[i] = [genre[i]]
    vals = [val for val in d.values()]
    merged = [descending_weighted_avg(x) for x in vals]
    return vals

def descending_weighted_avg(nums):
    n = len(nums)
    min_weight = (1 / n)
    weight = min_weight
    total_percent = 0
    avg = 0

    for i in range(n):
        avg += weight * nums[i]
        total_percent += weight
        weight =  (1 - total_percent) / n
    return avg / total_percent

genres_train = pd.read_csv('data_by_genres.csv',
names=["genres", "acousticness", "danceability",
 "duration", "energy", "instrumentalness", "liveness",
  "loudness", "speechiness", "tempo", "valence"]
)

genres_test = pd.read_csv('genres_test.csv',
names=["genres", "acousticness", "danceability",
 "duration", "energy", "instrumentalness", "liveness",
  "loudness", "speechiness", "tempo", "valence" ]
)
genres_test.pop('genres')

genres_features = genres_train.copy()
genres_labels = genres_features.pop('genres')
genres_class = [x[0] for x in genres_labels.keys()]
genres_labels = np.array([x for x in range(len(genres_labels))])
genres_features = np.array(genres_features)


normalize = preprocessing.Normalization()
normalize.adapt(genres_features)

model = tf.keras.models.Sequential([
    normalize,
    tf.keras.layers.Flatten(input_shape=(3232, 10)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(genres_class))
])



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(genres_features, genres_labels, epochs=15)

merged = merge_genres(genres_test.values)
predictions = model.predict(genres_test)

print(predictions)
print(genres_class[predictions[0].argmax()])    