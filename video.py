# make sure to have opencv-contrib-python aswell
import cv2
import numpy as np
import math
import os

# video initalize
vid = cv2.VideoCapture(0)

if not vid.isOpened():
    raise IOError("Cannot open")

# utils
def marker(image, center, color, LL=5):
	center = (int(center[0]), int(center[1]))
	image = cv2.line(image,
	(center[0] - LL, center[1]),
	(center[0] + LL, center[1]),
	color,
	3)
	image = cv2.line(image,
	(center[0], center[1] - LL),
	(center[0], center[1] + LL),
	color,
	3)
	return image


def pp(image, center, color):
	center = (int(center[0]), int(center[1]))
	image = cv2.line(image,
	(center[0], center[1]),
	(center[0], center[1]),
	color,
	3)
	image = cv2.line(image,
	(center[0], center[1]),
	(center[0], center[1]),
	color,
	3)
	return image

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(int(img_overlay.shape[0]), img.shape[0] - y)
    x1o, x2o = max(0, -x), min(int(img_overlay.shape[1]), img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

# video loop
while True:
     ret, frame = vid.read()
     image=frame

     h, w = frame.shape[:2]
     h,w = int(h),int(w)

     if not ret:
          print('something went wrong')
          break

     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
     mask = cv2.inRange(hsv, (40, 20, 20), (70, 255,255)) #(175,100,100),(180,255,255))#

     points = np.argwhere(mask == 255)
     npoi = [x for x in points if (x[0] > 200)]
     npoi = [y for y in points if (y[1] > 100)]
     #npoi = [p for p in points if math.dist([h/2,w/2],[p[0],p[1]])<200] # 300px radius around the center
     x = [p[1] for p in npoi]
     y = [p[0] for p in npoi]

     #mask = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

     for i in range(len(npoi)):
          image =  pp(image, [npoi[i][1],npoi[i][0]], (50, 205, 50))

     if len(npoi)>0:
          centroid = (sum(x) / len(npoi), sum(y) / len(npoi))
          #print(f"c1 {centroid}")
     else:
          centroid = [w/2,h/2]
     
     y = int(centroid[1])
     x = int(centroid[0])

     # the orb
     
     orbd = math.dist([h/2,w/2],[x,y])
     print(f"d {orbd}")
     if orbd>400:
         #assert os.path.exists("C:/Users/abhik/Desktop/video/orb5.png")
         orb = cv2.imread("C:/Users/abhik/Desktop/video/orb5.png",-1)
     if ((orbd > 300) and (orbd <600)):
         #assert os.path.exists("C:/Users/abhik/Desktop/video/orb4.png")
         orb = cv2.imread("C:/Users/abhik/Desktop/video/orb4.png",-1)
     if ((orbd > 100) and (orbd <300)):
         #assert os.path.exists("C:/Users/abhik/Desktop/video/orb3.png")
         orb = cv2.imread("C:/Users/abhik/Desktop/video/orb3.png",-1)
     if ((orbd > 50) and (orbd <100)):
         #assert os.path.exists("C:/Users/abhik/Desktop/video/orb2.png")
         orb = cv2.imread("C:/Users/abhik/Desktop/video/orb2.png",-1)
     if orbd < 50:
         #assert os.path.exists("C:/Users/abhik/Desktop/video/orb1.png")
         orb = cv2.imread("C:/Users/abhik/Desktop/video/orb1.png",-1)

     # Prepare inputs
     image = np.array(image)
     img_overlay_rgba = np.array(orb)

     # Perform blending
     alpha_mask = img_overlay_rgba[:, :, 3] / 255.0
     img_result = image[:, :, :3]
     img_overlay = img_overlay_rgba[:, :, :3]
     overlay_image_alpha(img_result, img_overlay, x, y, alpha_mask)


     
     h, w = frame.shape[:2]
     #print(h,w)
     mask = marker(mask, centroid, (203,192,255))
     image = marker(image, centroid, (203,192,255))
     image = marker(image, [w/2,h/2], (203,192,255))
     cv2.imshow("Tracking", mask)
     cv2.imshow("Original", image)

     key = cv2.waitKey(1)
     if key == 27: # escape key
          break

cv2.destroyWindow('frame')