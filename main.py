from flask import Flask
import requests
from utils import *


app = Flask(__name__)
@app.route('/')
def main():
    # Load models 
    style_predict_path = tf.keras.utils.get_file('style_predict.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite')
    style_transform_path = tf.keras.utils.get_file('style_transform.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite')
    
    # Get random dog image
    endpoint = "https://dog.ceo/api/breeds/image/random"
    response = requests.get(endpoint)
    img_url = response.json()['message']
    
    # Process the style and content images
    content_image = preprocess_image(load_img(img_url), 384)
    style_image = preprocessed_style_image = preprocess_image(load_img(style_img_url), 256)
    # Calculate style bottleneck for the preprocessed style image.
    style_bottleneck = run_style_predict(style_image)
    # Stylize the content image using the style bottleneck.
    stylized_image = run_style_transform(style_bottleneck, content_image)
    # Create a plot of the images
    plot(content_image, style_image, stylized_image)
    # Display the output in html
    image_html = """
    <h1>I hope you enjoy your random dog!</h1>
    <img src = "plot.png" alt = "Transformed Dog">
    """

    return (image_html)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
