


import glob


# count = 500;
# for filename in glob.iglob('data\Images\**\*.jpg', recursive=True):
	# print(filename)
	# count -= 1;
	# if(count < 0):
		# break;


	
all_images_fn = list(glob.iglob('data\Images\**\*.jpg', recursive=True));
print(len(all_images_fn))


























