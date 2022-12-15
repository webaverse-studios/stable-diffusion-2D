from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import base64

#------------------------------------------------

def img2b4(img: Image, format = 'PNG'):
    im_file = BytesIO()
    img.save(im_file, format=format)
    im_bytes = im_file.getvalue()  
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    return f'data:image/{format.lower()};base64,{im_b64}'   

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
    box_width = width / 3
    box_height = height / 3
    images = []
    for i in range(3):
        for j in range(3):
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

#------------------ summerstay's strongest component masking algo -------------------------------

def convertPILtocv2(im):
    cv2_im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return cv2_im

def convertcv2toPIL(cv2_im):
    pil_im = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB))
    return pil_im

def mask_from_black(PILpic, outer_tolerance=35, inner_tolerance=7):
    pic = convertPILtocv2(PILpic)
    (height, width, colors) = pic.shape

    # gray_img = cv2.cvtColor(pic , cv2.COLOR_BGR2GRAY)
    # threshold = cv2.threshold(gray_img, inner_tolerance, 255, cv2.THRESH_BINARY)[1]
    # analysis = cv2.connectedComponentsWithStats(threshold,4,cv2.CV_32S)
    # (totalLabels, label_ids, values, centroid) = analysis
    # output = np.zeros(gray_img.shape, dtype="uint8")
    # for i in range(1, totalLabels):
    #   # Area of the component
    #     area = values[i, cv2.CC_STAT_AREA]
    #     if (area > 4000):
    #         componentMask = (label_ids == i).astype("uint8") * 255
    # pic = cv2.bitwise_and(pic, pic, mask=componentMask)


    # place a tiny black square in each corner to remove any stray pixels
    pic[0:3,0:3,:]=(0,0,0)
    pic[height-3:height,0:3,:]=(0,0,0)
    pic[0:3,width-3:width,:]=(0,0,0)
    pic[height-3:height,width-3:width,:]=(0,0,0)

    #floodfill from the outside corners so that everything approximately black that is connected to the outside corners becomes completely black.
    ot = outer_tolerance
    cv2.floodFill(pic, None, (2,width-2), (0,0,0), (ot, ot, ot, ot), (ot, ot, ot, ot), cv2.FLOODFILL_FIXED_RANGE) 
    cv2.floodFill(pic, None, (height-2,2), (0,0,0), (ot, ot, ot, ot), (ot, ot, ot, ot), cv2.FLOODFILL_FIXED_RANGE) 
    cv2.floodFill(pic, None, (2,2), (0,0,0), (ot, ot, ot, ot), (ot, ot, ot, ot), cv2.FLOODFILL_FIXED_RANGE) 
    cv2.floodFill(pic, None, (height-2,width-2), (0,0,0), (ot, ot, ot, ot), (ot, ot, ot, ot), cv2.FLOODFILL_FIXED_RANGE)
     
    #make everything anywhere in the image that is nearly black completely black. This is usually done at a lower tolerance than the outer tolerance.
    lower = np.array([0, 0, 0], dtype="uint8")
    upper = np.array([inner_tolerance, inner_tolerance, inner_tolerance], dtype="uint8")
    cv2mask = cv2.inRange(pic, lower, upper)
    mask = convertcv2toPIL(cv2mask)
    return mask

def cutv2(img: Image, format = 'PNG', outer_tolerance=15, inner_tolerance=1):
  mask = mask_from_black(img, outer_tolerance=outer_tolerance, inner_tolerance=inner_tolerance)
  img = img.convert('RGBA')
  img_arr = np.array(img)
  img_arr[:,:,3] = 255 - np.array(mask.convert('L'))
  return Image.fromarray(img_arr, mode = 'RGBA')

