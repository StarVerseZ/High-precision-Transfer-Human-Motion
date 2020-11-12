from glob import glob
import cv2

imgs = []
imgs += glob('../datasets/vid2vid/train_src_img/*.jpg')
imgs += glob('../datasets/vid2vid/train_src_openpose/*.jpg')
imgs += glob('../datasets/vid2vid/train_src_densepose/*.png')
imgs += glob('../datasets/vid2vid/train_trg_img/*.jpg')
imgs += glob('../datasets/vid2vid/train_trg_openpose/*.jpg')
imgs += glob('../datasets/vid2vid/train_trg_densepose/*.png')
for i, img in enumerate(imgs):
    im = cv2.imread(img)
    im_crop = im[:, 128: 128+256, :]
    cv2.imwrite(img, im_crop)