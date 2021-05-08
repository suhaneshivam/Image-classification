import cv2
import numpy

def Conv2d(image ,kernal):

    (kh ,kw) = kernal.shape[:2]
    (ih ,iw) = image.shape[:2]

    pad = (kh -1) //2
    output = np.zeros((ih ,iw) ,dtype = "float32")
    image = cv2.copyMakeBorder(image ,pad ,pad ,pad ,pad ,cv2.BORDER_REPLICATE)

    for y in range(pad ,ih + pad):
        for x in range(pad , iw + pad):

            roi = image[y-pad:y+pad+1 ,x-pad:x+pad+1 ]
            value = (roi * kernal).sum()

            output[y-pad ,x-pad] = value
    return output
