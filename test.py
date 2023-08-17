import io
import requests
from PIL import Image
import base64

API_KEY = ""
#read test.txt file and get text
file = open("test.txt", 'r')
imgB64 = file.read()
file.close()

def loadImage(image : Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

#convert imageb64 to PIL Image
img = Image.open(io.BytesIO(base64.b64decode(imgB64)))
#load image to byte array
image_file_object = loadImage(img)

r = requests.post('https://clipdrop-api.co/remove-background/v1',
  files = {
    'image_file': ('car.jpg', image_file_object, 'image/jpeg'),
    },
  headers = { 'x-api-key': API_KEY}
)
if (r.ok):
  image = Image.open(io.BytesIO(r.content))

  filename_depth = f"dada_Depth.png"
  image.save(filename_depth)
else:
  r.raise_for_status()
  
