from flask import Flask, render_template_string
import streamlit as st

app = Flask(__name__)

HTML_TEMPLATE = "SDG's : GENDER EQUALITY"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SDG's Gender Equality</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }
       .container {
            max-width: 1200px;
            margin: auto;
            overflow: hidden;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-top: 30px;
            margin-bottom: 50px;
        }
       .gallery {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin: 0 -10px;
        }
       .gallery-item {
            flex: 1 0 31%;
            margin: 0 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        img {
            max-width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SDG's Gender Equality</h1>
        <div class="gallery">
            <div class="gallery-item">
                <img src="https://via.placeholder.com/350x200?text=Gender+Equality+1" alt="Gender Equality 1">
            </div>
            <div class="gallery-item">
                <img src="https://via.placeholder.com/350x200?text=Gender+Equality+2" alt="Gender Equality 2">
            </div>
            <div class="gallery-item">
                <img src="https://via.placeholder.com/350x200?text=Gender+Equality+3" alt="Gender Equality 3">
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/rekrutmen")
def rekrutmen():
    return st_app()

def st_app():
    # Your Streamlit app code here
    st.title("Rekrutmen Tanpa Bias")
    #... rest of your Streamlit app code...

if __name__ == "__main__":
    app.run(debug=True)
