import os
import sys

import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


def main(dirname):
    files = os.listdir(dirname)
    files = [f for f in files if f.endswith('.exr')]
    files = [os.path.join(dirname, f) for f in files]

    accum = None
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if accum is None:
            accum = img
        else:
            accum += img

    outname = os.path.join(dirname, 'accum.exr')
    cv2.imwrite(outname, accum)
    print('Write: %s' % (outname))

    # tone mapping
    tmo = cv2.createTonemapDrago(gamma=2.2)
    accum2 = tmo.process(accum)
    accum2 = (accum2 * 255.0).astype('uint8')

    outname = os.path.join(dirname, 'accum.png')
    cv2.imwrite(outname, accum2)
    print('Write: %s' % (outname))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("[usage] python accumulate.py IMAGE_FOLDER")
    else:
        main(sys.argv[1])
