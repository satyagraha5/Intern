import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.datasets import mnist
import keras 
from keras import backend as K

def set_args():
    args = {
        "model": "Schrodinger_GA",
        "centuries": 2,
        "population": 4,
        "survival_ratio": 0.5,
        "chromosome_length": 4,
        "mutation": 0.1,
        "cross_over": 0.5,
        "batch_size": 32,
        "epoch": 2,
        "validation_split": 0.25,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "loss": "binary_crossentropy",
        "metric": ["accuracy", precision_threshold, recall_threshold, f1_score_threshold]
    }
    return args

def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(y_pred)
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

def f1_score_threshold(threshold = 0.5):
    def f1_score(y_true, y_pred):
        recall_fn = recall_threshold(threshold)
        precision_fn = precision_threshold(threshold)
        recall = recall_fn(y_true, y_pred)
        precision = precision_fn(y_true, y_pred)
        f1_score_ratio = ( 2 * recall * precision) / (recall + precision + K.epsilon())
        return f1_score_ratio
    return f1_score

def genesis(args):
    earth = list()
    for i_th in range(args["population"]):
        creature = Schrodinger_GA(args, i_th)
        earth.append(creature)
    return earth

def struggle(args, earth, train_dataset, test_dataset):
    for i_th_century in range(args["centuries"]):
        print("=======================================")
        print("Century: {}".format(i_th_century))
        for i_th, creature in enumerate(earth):
            creature.fit(train_dataset)
            loss, acc, precision, recall, f1_score = creature.evaluate(test_dataset)
            creature.fitness = f1_score
            print(creature)
            print("\t\t   Loss: {:.3f} Acc: {:.3f} Precision: {:.3f} Recall: {:.3f} F1 Score: {:.3f}".format(loss, acc, precision, recall, f1_score))
        earth.sort(reverse = True)

        print("=======================================")
    return earth

def survival(args, earth):
    num_survived = round(len(earth) * args["survive_ratio"])
    for i_th in range(num_survived):
        
    return earth

def preprocessing_images(images):
    images = images.reshape(images.shape[0], 28 * 28)
    images = images/255.0
    return images

def preprocessing_labels(labels):
    new_labels = np.zeros_like(labels, dtype='float32')
    fail = [0]
    for i_th, label in enumerate(labels):
        if label in fail:
            new_labels[i_th] = 1.0
        else:
            new_labels[i_th] = 0.0
    return new_labels

class Schrodinger_GA():
    def __init__(self, args, name):
        self.name = name
        self.args = args
        self.model = keras.Sequential()
        self.birth_generation = 0
        self.fitness = 0
        self.chromosome = list()
        self._birth()
        self._build()

    def __repr__(self):
        return """ 
                   Creature Name:    {}
                   Birth Generation: {}
                   Fitness:          {:.3f}
                   Chromosome:       {}\n""".format(self.name, self.birth_generation, self.fitness, self.chromosome)

    def _birth(self):
        args = self.args
        chromosome = self.chromosome
        out_node_0 = round(random.random(), 3)
        out_node_1 = round(random.random(), 3)
        class_weight = round(random.random(), 3)
        threshold = round(random.random(), 3)
        chromosome.append(out_node_0)
        chromosome.append(out_node_1)
        chromosome.append(class_weight)
        chromosome.append(threshold)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def _build(self):
        model = self.model
        chromosome = self.chromosome
        eval("model.add(keras.layers.Dense({}, input_shape = (784, )))".format(int(20 * chromosome[0]) + 1))
        eval("model.add(keras.layers.BatchNormalization())")
        eval("model.add(keras.layers.Activation('relu'))")
        eval("model.add(keras.layers.Dense({}))".format(int(20 * chromosome[1]) + 1))
        eval("model.add(keras.layers.BatchNormalization())")
        eval("model.add(keras.layers.Activation('relu'))")
        eval("model.add(keras.layers.Dense(1, activation = 'sigmoid'))")
        metrics =  ["accuracy", precision_threshold(chromosome[3]), recall_threshold(chromosome[3]), f1_score_threshold(chromosome[3])]
        model.compile(optimizer = args["optimizer"], loss = args["loss"], metrics = metrics)

    def fit(self, train_dataset):
        model = self.model
        chromosome = self.chromosome
        model.fit(train_dataset["image"], train_dataset["label"], epochs = args["epoch"], batch_size = args["batch_size"], 
                  validation_split = args["validation_split"], class_weight = {0: chromosome[3], 1: 1.0}, verbose = 0)

    def evaluate(self, test_dataset):
        model = self.model
        return model.evaluate(test_dataset["image"], test_dataset["label"], verbose = 0)
    
    def predict(self, test_dataset):
        model = self.model
        return model.predict(test_dataset)

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    args = set_args()

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = preprocessing_images(training_images)
    training_labels = preprocessing_labels(training_labels)
    test_images = preprocessing_images(test_images)
    test_labels = preprocessing_labels(test_labels)
    train_dataset = {"image": training_images, "label": training_labels}
    test_dataset = {"image": test_images, "label": test_labels}

    # #In the Beginning, God Created Heaven and Earth
    earth = genesis(args)
    
    # #Struggle for existance
    earth = struggle(args, earth, train_dataset, test_dataset)