import glob
import cv2
import numpy as np


def decode_hadamard_scans(topleft_x=565, topleft_y=352, bottomright_x=992, bottomright_y=780, order=4, crop=0):
    topleft = np.array([topleft_x, topleft_y])
    bottomright = np.array([bottomright_x, bottomright_y])

    # Generate H
    H = np.array([[1]])
    for i in range(0, order):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

    if crop:
        # reshaping the images so that the cropped image encompasses correctly the hadamard code that is projected
        image_dimension_crop = abs(bottomright - topleft)
        filepath = glob.glob('/home/sreenithy/slope_LT/car_pos_img1/*.png')
        for f in filepath:
            x = cv2.imread(f)
            x = x[topleft_y:bottomright_y, topleft_x:bottomright_x]
            cv2.imwrite(f, x)

        filepath = glob.glob('/home/sreenithy/slope_LT/car_neg_img1/*.png')
        for f in filepath:
            x = cv2.imread(f)
            x = x[topleft_y:bottomright_y, topleft_x:bottomright_x]
            cv2.imwrite(f, x)

    else:
        img = cv2.imread('/home/sreenithy/slope_LT/car_neg_img1/1.png')
        image_dimension_crop = img.shape

    # Reshaping each channel of the captured image[(m,n,3)] to a vector of size (mn,1)
    lpos1 = []
    lpos2 = []
    lpos3 = []
    npos1 = []
    npos2 = []
    npos3 = []
    filepath = glob.glob('/home/sreenithy/slope_LT/car_pos_img1/*.png')
    for f in filepath:
        x = cv2.imread(f)
        [m, n, p] = np.shape(x)
        x1 = np.reshape(x[:, :, 0], (m * n, 1))
        x2 = np.reshape(x[:, :, 1], (m * n, 1))
        x3 = np.reshape(x[:, :, 2], (m * n, 1))
        lpos1.append(x1)
        lpos2.append(x2)
        lpos3.append(x3)

    filepath = glob.glob('/home/sreenithy/slope_LT/car_neg_img1/*.png')
    for f in filepath:
        x = cv2.imread(f)
        [m, n, p] = np.shape(x)
        x1 = np.reshape(x[:, :, 0], (m * n, 1))
        x2 = np.reshape(x[:, :, 1], (m * n, 1))
        x3 = np.reshape(x[:, :, 2], (m * n, 1))
        npos1.append(x1)
        npos2.append(x2)
        npos3.append(x3)


    [bb, aa, _] = np.shape(lpos1)

    lpos1 = np.transpose(np.reshape(lpos1, (bb, aa)))
    lpos2 = np.transpose(np.reshape(lpos2, (bb, aa)))
    lpos3 = np.transpose(np.reshape(lpos3, (bb, aa)))

    npos1 = np.transpose(np.reshape(npos1, (bb, aa)))
    npos2 = np.transpose(np.reshape(npos2, (bb, aa)))
    npos3 = np.transpose(np.reshape(npos3, (bb, aa)))

    ll = np.empty((aa, bb, 3), dtype=float)
    ll[:, :, 0] = lpos1
    ll[:, :, 1] = lpos2
    ll[:, :, 2] = lpos3

    nn = np.empty((aa, bb, 3))
    nn[:, :, 0] = npos1
    nn[:, :, 1] = npos2
    nn[:, :, 2] = npos3
    c = ll - nn

    Hpos = np.transpose(H)

    for i in range(16):
        for j in range(16):
            if Hpos[i, j] < 0:
                Hpos[i, j] = 0

    T = np.empty((aa, bb, 3))
    T[:, :, 0] = c[:, :, 0] @ np.linalg.inv(np.transpose(Hpos))
    T[:, :, 1] = c[:, :, 1] @ np.linalg.inv(np.transpose(Hpos))
    T[:, :, 2] = c[:, :, 2] @ np.linalg.inv(np.transpose(Hpos))

    np.save("Tmatrix", T)

    # Relighting
    x1 = np.empty((aa, 1, 3))
    x1[:, :, 0] = T[:, :, 0] @ np.ones((16, 1)) * 1
    x1[:, :, 1] = T[:, :, 1] @ np.ones((16, 1)) * 0
    x1[:, :, 2] = T[:, :, 2] @ np.ones((16, 1)) * 1

    x1 = np.reshape(x1, (image_dimension_crop[0], image_dimension_crop[1], 3))
    x1 = x1[:, :, ::-1]
    cv2.imwrite("relit.png", x1)


decode_hadamard_scans()
