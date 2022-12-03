from PIL import Image
from io import BytesIO
import base64

def splitHeightTo2(img: Image):
    width, height = img.size
    box_count = 2
    box_height = height / box_count
    print(box_height)
    images = []
    for i in range(box_count):
        box = (0, box_height * i, width, box_height * (i + 1))
        a = img.crop(box)
        images.append(img2b4(a))
    return images

def splitImageTo9(img: Image):
    width, height = img.size
    box_count = 3
    box_width = width / box_count
    box_height = height / box_count
    images = []
    for i in range(box_count):
        for j in range(box_count):
            box = (box_width * j, box_height * i, box_width * (j + 1), box_height * (i + 1))
            a = img.crop(box)
        images.append(img2b4(a))
    return images


def cut(img: Image, format = 'PNG'):
    img = img.convert('RGBA')

    width, height = img.size
    pixels = img.getcolors(width * height)
    most_frequent_pixel = pixels[0]

    for count, colour in pixels:
        if count > most_frequent_pixel[0]:
            most_frequent_pixel = (count, colour)

    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))
            if abs(pixel[0] - most_frequent_pixel[1][0]) < 10 and abs(pixel[1] - most_frequent_pixel[1][1]) < 10 and abs(pixel[2] - most_frequent_pixel[1][2]) < 10:
                img.putpixel((x, y), (255, 255, 255, 0))
                
    return img



def img2b4(img: Image, format = 'PNG'):
    im_file = BytesIO()
    img.save(im_file, format=format)
    im_bytes = im_file.getvalue()  
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    return f'data:image/{format.lower()};base64,{im_b64}'   