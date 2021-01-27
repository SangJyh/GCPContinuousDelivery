from flask import Flask
import requests

app = Flask(__name__)
@app.route('/')
def main():
    endpoint = "https://dog.ceo/api/breeds/image/random"
    response = requests.get(endpoint)
    img_url = response.json()['message']
    image_html = """
    <h1>I hope you enjoy your random dog!</h1>
    <img src = """ + img_url + """ alt = "Random dog" style="width:500px;height:500px;">
    """

    return (image_html)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
