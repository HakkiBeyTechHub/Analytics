import json
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import base64
import streamlit as st
import pandas as pd

# background image to streamlit
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = (
    """
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: scroll; # doesn't work
        }
        </style>
    """
        % bin_str
    )

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


file = st.file_uploader("Ahşap Görseli Yükle", key="file_uploader")
if file is not None:
    try:
        img = plt.imread(file).reshape(1, 200, 200, 3) / 255
        st.image(img, use_column_width=True)
    except Exception as exc:
        print(exc)
        st.error(
            "Görsel uygun değil JPG ya da PNG uzantılı dosyalar yüklemeyi deneyin."
        )


def load_model():
    model = tf.keras.models.load_model(
        "75acc.h5"
    )
    return model


def predict():
    pred = m.predict(img)
    print(np.round(pred, 2))
    pred_prob = str(round(np.max(pred) * 100, 2))
    pred = np.argmax(pred, axis=-1)[0]
    pred = label_map[pred]
    result = load_info_card(pred)
    st.metric(
        "Ahşap Sınıfı: ",
        result["name"],
        delta="%" + pred_prob,
        delta_color="off",
        help=None,
        label_visibility="visible",
    )
    result.pop("name")
    result.update({"file_number": pred})
    st.write(pd.DataFrame.from_dict(result, orient="index").rename(columns={0: "Info"}))

def load_class_indices():
    with open("class_indices.json", "r") as rd:
        label_dict = json.load(rd)
        label_map = dict((v, k) for k, v in label_dict.items())
    return label_map

def load_info_card(label):
    with open("wood_info.json", "r") as rd:
        info_dict = json.load(rd)
        label_info_dict = info_dict[label]
    return label_info_dict

m = load_model()
label_map = load_class_indices()
pred = st.button("Sınıflandır", key=None, help=None, type="primary", disabled=False)
if pred:
    predict()
