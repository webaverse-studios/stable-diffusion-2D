from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import base64
from rembg import remove

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

def mask_from_black(generated_image, init_image, outer_tolerance=35, inner_tolerance=7, radius = 70):
    pic = convertPILtocv2(generated_image)
    init_pic = convertPILtocv2(init_image)
    grey_mask = cv2.cvtColor(init_pic, cv2.COLOR_BGR2GRAY)
    ret, starting_mask = cv2.threshold(grey_mask, 0, 255, cv2.THRESH_BINARY_INV)
    (height, width, colors) = pic.shape
    gray_img = cv2.cvtColor(pic , cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray_img, inner_tolerance, 255, cv2.THRESH_BINARY)[1]
    analysis = cv2.connectedComponentsWithStats(threshold,4,cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    output = np.zeros(gray_img.shape, dtype="uint8")
    biggest_area = 0
    biggest_index = 0
    for i in range(1, totalLabels):
      # Area of the component
        area = values[i, cv2.CC_STAT_AREA]
        if area>biggest_area:
            biggest_area = area
            biggest_index = i
        
    componentMask = (label_ids == biggest_index).astype("uint8") * 255
    pic = cv2.bitwise_and(pic, pic, mask=componentMask)

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
    #keep the center of the original mask and only affect the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    eroded = cv2.erode(starting_mask,kernel)
    cv2mask = cv2.bitwise_or(cv2mask,eroded)
    dilated =cv2.dilate(starting_mask, kernel)
    cv2mask = cv2.bitwise_and(cv2mask,dilated)
    mask = convertcv2toPIL(cv2mask)
    # cv2.imshow("mask",cv2mask)
    # cv2.waitKey(0)
    return mask

def cutv2(generated_image:Image, init_image:Image, format = 'PNG', outer_tolerance=35, inner_tolerance=7, radius = 70):
  mask = mask_from_black(generated_image, init_image, outer_tolerance=outer_tolerance, inner_tolerance=inner_tolerance, radius = radius)
  generated_image = generated_image.convert('RGBA')
  img_arr = np.array(generated_image)
  img_arr[:,:,3] = 255 - np.array(mask.convert('L'))
  return Image.fromarray(img_arr, mode = 'RGBA')

def create_foreground_mask(input_im: Image, init_pic = None, radius = 30):
    
    # I don't know how to convert to the right format so I save it and reopen it
    output_path = 'out.png'
    input_path = "in.png"
    input_im.save(input_path)
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            # perform background subtraction
            input = i.read()
            output = remove(input)
            o.write(output)
            im = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
            # the alpha channel is the mask. We want one pixel less of foreground, to eliminate all the antialiased pixels
            mask = im[:,:,3]
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode (mask, kernel)
            # now we combine it with the mask from the init image.
            if init_pic != None:
                init_pic_cv2 = convertPILtocv2(init_pic)
                grey_mask = cv2.cvtColor(init_pic_cv2, cv2.COLOR_BGR2GRAY)
                ret, init_mask = cv2.threshold(grey_mask, 0, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
                dilated = cv2.dilate(init_mask,kernel)
                mask = cv2.bitwise_and(mask,dilated)
                eroded =cv2.erode(init_mask, kernel)
                mask = cv2.bitwise_or(mask,eroded)
            out = convertcv2toPIL(mask)
            return out

def cutv3(generated_image:Image, init_image:Image, radius = 30):
  # mask = mask_from_black(generated_image, init_image, outer_tolerance=outer_tolerance, inner_tolerance=inner_tolerance)
  mask = create_foreground_mask(generated_image,init_image,radius)
  generated_image = generated_image.convert('RGBA')
  img_arr = np.array(generated_image)
  img_arr[:,:,3] = np.array(mask.convert('L'))
  return Image.fromarray(img_arr, mode = 'RGBA')