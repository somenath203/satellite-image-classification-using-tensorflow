# Satellite Image Classifier

## Introduction
The aim of this project is to classify satellite images into their respective categories i.e. 'Cloudy', 'Desert', 'Green Area' and 'Water' using Convolutional Neural Networks (CNNs) implemented with TensorFlow.

## Dataset used in this project

The dataset used in this project is taken from kaggle: https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification

## Models used in this project

The model used for prediction is **Pre-trained resnet101 model**. The training accuracy of the model is around **99.87%** and the testing accuracy is around **99.50%**.

## About the web application of the deep learning model

The deep learning model of this project is connected with an application created with Gradio for real time prediction and it is deployed on HuggingFace Spaces.

## Links

Live Preview: https://huggingface.co/spaces/som11/satellite_image_classification

## Warning
While the model of this project can classify images correctly, but in some cases, the model may misclassify the images, therefore, it is strongly advised not to rely solely on the output of this model.
