import json
import random
import tensorflow
from tensorflow.python.ops.gen_parsing_ops import parse_single_sequence_example
import tflearn
import numpy
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

# load intents .json
with open("intents.json") as file:
    data = json.load(file)

# load pickle (after execute the training.py)
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)


# Model
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 9)
net = tflearn.fully_connected(net, 9)
net = tflearn.fully_connected(net, 9)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.load("model.tflearn")


def bag_of_words(sentance, words):
    bag = [0 for _ in range(len(words))]
    sentance_words = nltk.word_tokenize(sentance)
    sentance_words = [stemmer.stem(word.lower()) for word in sentance_words]

    for se in sentance_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = (1)

    return numpy.array(bag)


def chat(user_input):
    results = model.predict([bag_of_words(user_input, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    print("Probabilities are : ", results)
    if results[results_index] > 0.7:
        # make random response
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        return(random.choice(responses))
    else:
        return ("I don't quite understand. Try again or ask a diffrent question.")


# Call chat function
chat("Hey")
