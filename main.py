from flask import Flask
from flask import jsonify, make_response
import requests

app = Flask(__name__)
@app.route('/')
def main():
    endpoint = "https://dog.ceo/api/breeds/image/random"
    response = requests.get(endpoint)
    img_url = response.json()['message']
    image_html = """<img src = """ + img_url + """ alt = "Random Basenji Image">"""
    return (image_html)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
