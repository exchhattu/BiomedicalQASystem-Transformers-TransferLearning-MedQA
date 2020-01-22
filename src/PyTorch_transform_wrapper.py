"""
Rojan Shrestha, Insight Data Science
Tue Jan 21 21:36:32 2020
"""

from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)


class PyTorchTransformWrapper:

    def __init__(self):
        self._qa_data = []

    def read_squad_formatted_json_file(self, path_to_file): 
        self._qa_data = read_squad_examples(input_file=path_to_file, is_training=True, version_2_with_negative=False)
