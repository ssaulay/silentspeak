import requests

url = "http://127.0.0.1:8000/predict"
response = requests.post(url=url, files={"file": open('/home/clement/code/ssaulay/silentspeak/drafts/data/sample_data/test_video/001_L01.avi', "rb")})
print(response.content)
