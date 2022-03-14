# Arabic-Dialect-Dataset-Classification

Goal:
        The goal is to build a model (Machine Learning and Deep Learning model) that predicts the dialect of a given the Arabic text as each country has its own dialect.

Dataset:
        A dialect dataset that consists of two main columns, Id and dialect. The “id” column is used to retrieve the text to be classified using POST request. This dataset                 contains 458197 samples.

Resources Problem:
        My resources (RAM) was not enough so I applied the model on google colab with 12 GB RAM, but I faced the same problem because of the memory available.
        Therefore, I took a fraction from the data, but this affected the accuracy.
        I have used many machine learning algorithms for classification and two models of deep learning but this did not improve the accuracy except for a little bit.
        I have made the model more complex (to solve the underfitting) but I was limited to resources available.
        I have used feature selection methods and dimensionality reduction methods, and I have removed non-arabic words from the sentences.
