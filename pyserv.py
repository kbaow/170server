from flask import Flask, request, after_this_request, jsonify
import json

app = Flask(__name__)

# Route to handle POST requests and output data
@app.route('/submit', methods=['POST'])
def submit():
    # Check if data is JSON or form data
    y = request.get_data()
    data_dict = json.loads(y)

    print(data_dict)

    return "hi"

if __name__ == "__main__":
    app.run(port=8000, debug=True)