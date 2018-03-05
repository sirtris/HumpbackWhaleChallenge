
def import_RGB(location):
    im = Image.open(location)
    #im.show()
    return im

def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')

if (__name__ == '__main__'):

    test = import_RGB('data/train/00a29f63.jpg')
    test2 = test.convert('HSV' )#Convert image to HSV

    #CHANGE HUE
    #layer = Image.new('RGB', test2.size, 'green') # "hue" selection is done by choosing a color...
    #test3 = Image.blend(test2, layer, 0.1)

    #CHANGE CONTRAST


    #test
    test3.show()
    print(test2)
    #test2.save('test.png')
