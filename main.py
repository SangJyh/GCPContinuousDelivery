from flask import Flask
import requests
from utils import load_img, plot, preprocess_image, run_style_predict, run_style_transform
import base64
from io import BytesIO

app = Flask(__name__)
@app.route('/')
def main():
    #Get random dog image 
    endpoint = "https://dog.ceo/api/breeds/image/random"
    response = requests.get(endpoint)
    img_url = response.json()['message']
    # Get style image and process it 
    style_img_url = "https://upload.wikimedia.org/wikipedia/en/1/14/Picasso_The_Weeping_Woman_Tate_identifier_T05010_10.jpg"
    style_image = preprocess_image(load_img(style_img_url), 256)
    #Calculate style bottleneck for the preprocessed style image
    style_bottleneck = run_style_predict(style_image)

    #Process the content image
    content_image = preprocess_image(load_img(img_url), 384)
    # Stylize the content image using the style bottleneck
    stylized_image = run_style_transform(style_bottleneck, content_image)

    # Create a plot of the images
    fig = plot(content_image, style_image, stylized_image)
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')

    # Display the output in html
    image_html = '<h1>A randoml dog in the style of Picasso!</h1>' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    return (image_html)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
