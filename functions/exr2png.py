import cv2
import numpy as np

def convert_exr2png(exr_filename, png_filename, algo=None):
    image_exr = cv2.imread(exr_filename, -1)
    if algo is None:
        # Do nothing
        image_png = image_exr
    elif algo == 'gamma':
        # gamma correction
        image_png = np.power(np.clip(image_exr, 0.0, 1.0), 1.0 / 2.2)
    elif algo == 'durand':
        tmo = cv2.createTonemap(gamma=2.2)
        image_png = tmo.process(image_exr.copy())
    else:
        raise RuntimeError('Unknown TMO method: %s' % (algo))

    image_png = np.clip((255*image_png), 0, 255).astype(np.uint8)
    cv2.imwrite(png_filename, image_png)