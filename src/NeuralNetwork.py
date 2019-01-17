import pandas as pd


class NeuralNetwork:
    def __init__(self, input_width, input_height, classes):
        self.input_width = input_width
        self.input_height = input_height
        self.classes = classes
        self.model = {}

    def save(self, model_path, history_path):
        self.model.save(model_path, overwrite=True)
        history_frame = pd.DataFrame.from_dict(self.history.history)
        history_frame.to_csv(history_path)

    def load(self, path):
        self.model.load_weights(path)
