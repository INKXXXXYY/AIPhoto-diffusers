import requests
import json

url = "http://region-3.seetacloud.com:22056"

payload = {
  "prompt":"1girl"
}


headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    #"Cookie":"Hm_lvt_2bad21c50f160413ff22e3b46fe3b2f0=1693384680,1694073525,1694141403; access-token-unsecure=bf9nLobjFPNIgPNMtlzYTg",
}

response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload,headers=headers)

print(response.text)
