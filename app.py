import zipfile
import os
import gradio as gr
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.utils import img_to_array
from keras.applications.resnet import preprocess_input



zip_path = 'resnet101_model.zip'

extract_dir = 'resnet101_model'


if not os.path.isdir(extract_dir):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        zip_ref.extractall(extract_dir)


model = load_model(os.path.join(extract_dir, 'resnet101_model'))


class_names = ['cloudy', 'desert', 'green_area', 'water']


def predict_image(file_path):

    if file_path is None:

        return "Please upload a satellite image for prediction"
    

    with open(file_path, "rb") as f:

        imageUploadedByUser = Image.open(f)

        imageUploadedByUser = imageUploadedByUser.resize((224, 224))

        if imageUploadedByUser.mode != 'RGB':
            imageUploadedByUser = imageUploadedByUser.convert('RGB')

        image_to_arr = img_to_array(imageUploadedByUser)

        image_to_arr_preprocess_input = preprocess_input(image_to_arr)

        image_to_arr_preprocess_input_expand_dims = np.expand_dims(
            image_to_arr_preprocess_input, axis=0)

        prediction = model.predict(
            image_to_arr_preprocess_input_expand_dims)[0]

        prediction_argmax = np.argmax(prediction)

        prediction_final_result = class_names[prediction_argmax]

        return f'The predicted satellite image is {prediction_final_result}.'


custom_css = """
    .desc { text-align: center; }
"""


interfaceOfGradio = gr.Interface(
    fn=predict_image,
    inputs=gr.File(type="filepath",
                   label="Upload Satellite Image", file_count="single"),
    outputs="text",
    description="<p class='desc'>" +
    "Upload a satellite image on the left and see the prediction result on the right." + "</p>",
    css=custom_css
)


interfaceOfGradio.launch()
