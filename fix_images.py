import glob
import cv2
import os
import sys

def resize_crop(img, x):
	#crop img into x*x square shape
	h = img.shape[0]
	w = img.shape[1]
	if(h<w):
		ratio = float(w)/h
		new_h = x
		new_w = int(x * ratio)
		crop_h0 = 0
		crop_h1 = x
		crop_w0 = int((new_w - x) / 2)
		crop_w1 = crop_w0 + x
	else:
		ratio = float(h)/w
		new_h = int(x * ratio)
		new_w = x
		crop_w0 = 0
		crop_w1 = x
		crop_h0 = int((new_h - x) / 2)
		crop_h1 = crop_h0 + x

	img = cv2.resize(img,(new_w,new_h),interpolation = cv2.INTER_AREA)
	img = img[crop_h0:crop_h1,crop_w0:crop_w1]
	return img

def main():
	if(len(sys.argv) > 2):
		cropsize = int(sys.argv[1])
		directory = sys.argv[2]
	else:
		print("arguments: cropsize, output directory name")
		sys.exit()

	if not os.path.exists(directory):
		os.makedirs(directory)

	all_images_fn = list(glob.iglob('data/Images/**/*.jpg', recursive=True))
	print(len(all_images_fn))

	for i,fn in enumerate(all_images_fn):
		img = cv2.imread(fn)
		img = resize_crop(img,cropsize)
		cv2.imwrite(directory+"/"+str(i)+".png",img)
		print("\r"+str(i)+"/"+str(len(all_images_fn)),end="\r")

if __name__=="__main__":
	main()



















