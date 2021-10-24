import requests

resp = requests.post("https://localhost:5000/predict", files={'file': open('images/xpd.jpg', 'rb')})

print(resp.text)