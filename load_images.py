import glob
import cv2

def resize_crop(img, x):
	h = img.shape[0];
	w = img.shape[1];
	
	if(h<w):
		ratio = float(w)/h;
		new_h = x;
		new_w = int(x * ratio);
	else:
		ratio = float(h)/w;
		new_h = int(x * ratio);
		new_w = x;
		
	img = cv2.resize(img,(new_w,new_h),interpolation = cv2.INTER_AREA);
		
	return img;

# count = 500;
# for filename in glob.iglob('data\Images\**\*.jpg', recursive=True):
	# print(filename)
	# count -= 1;
	# if(count < 0):
		# break;


	
all_images_fn = list(glob.iglob('data\Images\**\*.jpg', recursive=True));
print(len(all_images_fn))


# for fn in all_images_fn:

	


img = cv2.imread(all_images_fn[0])
img = resize_crop(img,30);

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# cv2.imwrite('messigray.png',img)


















