import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO

class Utils:
    @staticmethod
    def read_image(target_file):
        # img = plt.imread(target_file).reshape(1, 200, 200, 3) / 255
        pil_image = Image.open(BytesIO(target_file))
        pil_image = tf.keras.utils.img_to_array(pil_image).reshape(1, 200, 200, 3) / 255
        return pil_image

class Predictor:
    def __init__(self) -> None:
        self.model = self._load_model()
        self._label_map = self._load_class_indices()

    def _load_model(self):
        model = tf.keras.models.load_model(
            "75acc.h5"
        )
        return model
    
    def _load_class_indices(self):
        with open("class_indices.json", "r") as rd:
            label_dict = json.load(rd)
            label_map = dict((v, k) for k, v in label_dict.items())
        return label_map
    
    def _load_info_card(self, label):
        with open("wood_info.json", "r") as rd:
            info_dict = json.load(rd)
            label_info_dict = info_dict[label]
            label_info_dict.update({"file_number": label})
        return label_info_dict
    
    def predict(self, img):
        pred = self.model.predict(img)
        print(np.round(pred, 2))
        pred_prob = str(round(np.max(pred) * 100, 2))
        pred = np.argmax(pred, axis=-1)[0]
        pred = self._label_map[pred]
        result = self._load_info_card(pred)
        return result

    