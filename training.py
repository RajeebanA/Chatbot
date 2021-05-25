import json
import random
from nltk.util import pr
import tensorflow
from tensorflow.python.ops.gen_parsing_ops import parse_single_sequence_example
import tflearn
import numpy
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# load intents .json
with open("intents.json") as file:
    data = json.load(file)

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # Training Data
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Define Model
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 9)
net = tflearn.fully_connected(net, 9)
net = tflearn.fully_connected(net, 9)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Fit & Save the Model
model.fit(training, output, n_epoch=1000, batch_size=9, show_metric=True)
model.save("model.tflearn")

print("Model Successfully Saved🏳‍🌈")
