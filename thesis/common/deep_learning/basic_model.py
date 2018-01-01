import tensorflow as tf


class BasicModel:
    """
    Basic model on deep neural network
    """
    def __init__(self, input=None, labels=None, dims_in=None, dims_out=None):
        """
        :param input: input placeholder
        :param labels: label placeholder
        :param dims_in: input dimensions
        :param dims_out: output dimensions
        :return:
        """
        self.input = input
        self.labels = labels
        self.dims_in = dims_in
        self.dims_out = dims_out

    def model(self):
        return None

    def loss(self):
        return None

    def train(self):
        return None

    def evaluation(self):
        return None
