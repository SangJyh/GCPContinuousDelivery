from flask import Flask
from flask import jsonify 
import json 
from urllib.request import Request, urlopen

app = Flask(__name__)
@app.route('/')
def dashboard():
    req = Request("https://poetrydb.org/title/Ozymandias/lines.json")
    data = urlopen(req).read()
    return(data)
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
