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

# def convertcv2toPIL(cv2_im):
#     pil_im = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB))
#     return pil_im

def convertcv2toPIL(cv2_im):
    (height, width, colors) = cv2_im.shape
    if colors == 4:
        pil_im = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGRA2RGBA))
    else:
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

def cut_magenta(input_pil, outer_tolerance = 80):
    # input_image = cv2.imread(input_filename)
    input_image = convertPILtocv2(input_pil)

    (height, width, colors) = input_image.shape
    # convert the magenta background image into a black background image by inverting the red and blue channels.
    input_image[:,:,0] = 255-input_image[:,:,0]
    input_image[:,:,2] = 255-input_image[:,:,2]
    # the opacity is a clipped linear function of the maximum of the green channel and the inverted red and vlue channels.
    opacity = np.clip(2.0*np.amax(np.double(input_image),2)-64,0,255)
    transparent_array = np.zeros((input_image.shape[0],input_image.shape[1],4),np.uint8)
    transparent_array[:,:,:3]=input_image
    transparent_array[:,:,3]=opacity
    im = transparent_array
    # now reinvert the colors so that they are back to normal
    im[:,:,2] = 255 - im[:,:,2]
    im[:,:,0] = 255 - im[:,:,0]
    # next make a new transparency mask that is just for the surrounding backhground using floodfill
    ot = outer_tolerance
    outer_mask = np.zeros((input_image.shape[0]+2,input_image.shape[1]+2,1),np.uint8)
    for x in range(2,width-2,20):
        for y in [2,height-2]:
            if input_image[y,x,0]>0: 
                cv2.floodFill(input_image, outer_mask, (x,y), (0,0,0), (ot, ot, ot, ot), (ot, ot, ot, ot), cv2.FLOODFILL_FIXED_RANGE)
    for y in range(2,height-2,20):
        for x in [2,width-2]:
            if input_image[y,x,0]>0: 
                cv2.floodFill(input_image, outer_mask, (x,y), (0,0,0), (ot, ot, ot, ot), (ot, ot, ot, ot), cv2.FLOODFILL_FIXED_RANGE)  
    outer_mask = 255*(1-outer_mask)
    #take the minimum brightness (bitwise and) of the two methods to get the final transparency.
    im[:,:,3] = cv2.bitwise_and(im[:,:,3],outer_mask[1:-1,1:-1])
    #finally resize it to 128x128 and then upscale it again to make it pixelated.
    small=cv2.resize(im,(128,128))
    small=cv2.resize(small,(512,512),interpolation=cv2.INTER_NEAREST)
    #at this point you could convert it to a PIL image, but I haven't really tested that.
    # cv2.imwrite(input_filename + "_small.png", small)
    return convertcv2toPIL(small)


def cutv2(generated_image:Image, init_image:Image, format = 'PNG', outer_tolerance=35, inner_tolerance=7, radius = 70):
  mask = mask_from_black(generated_image, init_image, outer_tolerance=outer_tolerance, inner_tolerance=inner_tolerance, radius = radius)
  generated_image = generated_image.convert('RGBA')
  img_arr = np.array(generated_image)
  img_arr[:,:,3] = 255 - np.array(mask.convert('L'))
  return Image.fromarray(img_arr, mode = 'RGBA')

#------------Magenta ControlNet Canny Pixel model----------------

def convertPILtocv2(im):
    cv2_im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return cv2_im

def convertcv2toPIL(cv2_im):
    (height, width, colors) = cv2_im.shape
    if colors == 4:
        pil_im = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGRA2RGBA))
    else:
        pil_im = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB))
    return pil_im

def make_background_magenta(PILforeground_source, PILbackground_source, erode_width):
    # take a background source that has a pure magenta background, and replace the background on the foreground source with it.
    # this only works well if the background source was used with controlnet to generate the foreground source.
    # erode_width is how much the background should contract before being applied
    foreground_source = convertPILtocv2(PILforeground_source)
    background_source = convertPILtocv2(PILbackground_source)
    (height, width, colors) = foreground_source.shape
    print(foreground_source.shape)
    background_source = cv2.resize(background_source,(width, height), interpolation=cv2.INTER_LINEAR)
    mask =  np.all(background_source == [255,0,255], axis=-1)
    print(mask.dtype)
    if erode_width>0:
        kernel = np.ones((erode_width, erode_width), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8),kernel)
    else:
        kernel = np.ones((-erode_width, -erode_width), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8),kernel)
    foreground_source[mask.astype(bool)] = [255,0,255]

    # opacity = np.where(foreground_source == [255,0,255])
    # print(opacity[0].shape)
    transparent_array = np.zeros((foreground_source.shape[0],foreground_source.shape[1],4),np.uint8)
    transparent_array[:,:,:3]=foreground_source
    transparent_array[:,:,3]= np.where(foreground_source == [255,0,255], [0], [255])[:,:,0]

    print(transparent_array.shape)

    output = convertcv2toPIL(transparent_array)
    return output