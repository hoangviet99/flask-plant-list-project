# Flask API
Flask API for recorgnize image using Deep Learning model

## Step by step to deploy

### 1. Clone from git
    git clone https://github.com/hoangviet99/flask-plant-list-project.git
    cd flask-plant-list-project/

### 2. Create enviroment and import libraries
    python -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt

### 3. Starting serve
    cd app/
    export FLASK_APP=main.py
    export FLASK_ENV=development
    flask run
<br>
It will be run at http://127.0.0.1:5000/

### 4. Testing servee
Open another terminal in project folder then do following steps
    . venv/bin/activate
    cd test/
    python test.py

### End