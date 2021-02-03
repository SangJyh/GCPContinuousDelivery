import os
import tensorflow as tf
from skimage import io
import requests
import matplotlib.pyplot as plt

def load_img(img_url):
  """
  Function to load an image from a file, and add a batch dimension.
  Inputs:
  img_url: The url to load the file from.

  Outputs:
  img: The loaded image with an additional batch dimension.
  """
  img = io.imread(img_url)
  img = tf.convert_to_tensor(img)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]
  return img

def preprocess_image(image, target_dim):
  """
  Function to pre-process an image by resizing and central cropping it.
  Inputs:
  image: The image to be processed.
  target_dim: The target dimension of the image.

  Outputs:
  image: The resized and cropped image. 
  """
  # Resize the image so that the shorter dimension becomes 256px.
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  image = tf.image.resize(image, new_shape)

  # Central crop the image.
  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

  return image


def run_style_predict(preprocessed_style_image):
  """
  Function to run style prediction on preprocessed style image.
  Inputs:
  preprocessed_style_image: The image you want to extract style from.

  Outputs:
  style_bottleneck: A vector containing information about the style of the image. 
  """
  # Load the model.
  style_predict_path = tf.keras.utils.get_file('style_predict.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite')
  interpreter = tf.lite.Interpreter(model_path=style_predict_path)

  # Set model input.
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

  # Calculate style bottleneck.
  interpreter.invoke()
  style_bottleneck = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return style_bottleneck

def run_style_transform(style_bottleneck, preprocessed_content_image):
  """
  Run style transform on the preprocessed style image.

  Inputs:
  style_bottleneck: A vector containing information about the style of the image.
  preprocessed_content_image: The transformed image you want to stylize.

  Outputs:
  stylized_image: The image with the new style applied.
  """
  # Load the model.
  style_transform_path = tf.keras.utils.get_file('style_transform.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite')
  interpreter = tf.lite.Interpreter(model_path=style_transform_path)

  # Set model input.
  input_details = interpreter.get_input_details()
  interpreter.allocate_tensors()

  # Set model inputs.
  interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
  interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
  interpreter.invoke()

  # Transform content image.
  stylized_image = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()
  return(stylized_image)

def plot (content_image, style_image, stylized_image):
    """
    Plot the raw image and the newly stylized image. 
    
    Inputs:
    content_image: The raw image that you transformed.
    style_image: The image you extracted style from.
    stylized_image: The transformed image with the newstyle.

    Outputs:
    fig: A figure displaying the above images. 
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs[0].imshow(tf.squeeze(content_image))
    axs[0].axis('off')
    axs[0].set_title("Raw Image")
    axs[1].imshow(tf.squeeze(stylized_image))
    axs[1].axis('off')
    axs[1].set_title("Stylized Image")
    return (fig)