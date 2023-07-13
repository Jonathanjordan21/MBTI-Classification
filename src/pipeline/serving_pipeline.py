# Flask REST API Pipeline, ongoing...

from flask import Flask, request, jsonify, render_template
import pickle, os, json
# from src.utils import load_object

app = Flask(__name__)

@app.route('/pred', methods=['GET'])
def pred():
    model = None
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    with open(os.path.join(parent_dir, 'artifacts', 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    # model = load_object(os.path.join('artifacts', 'model.pkl'))
    comment = str(request.args.get('comment'))
    prediction = model.predict(comment)
    return jsonify(
        {
            "Your Comment" : comment,
            "Your MBTI Personality Type" : prediction 
        }
    )

@app.route('/')
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)