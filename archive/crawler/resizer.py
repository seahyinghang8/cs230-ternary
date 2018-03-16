from glob import glob
import os
import os.path
from PIL import Image

SIZE = 28, 28
# set directory
directory = '/home/ubuntu/yinghang/image-crawler/datasets/'
alpha = os.listdir(directory)
print(directory)

for line in alpha:
    if (line != 'a.out'):
        os.chdir(directory + line)
        # filter all jpg and png images
        IMAGE_FILES = glob('*.jpg')
        IMAGE_FILES.extend(glob('*.jpeg'))
        IMAGE_FILES.extend(glob('*.png'))
        IMAGE_COUNTER = 1
        resized_path = os.path.join('../../resized', line)
	
        if (os.path.exists(resized_path)):
            print("%s is already resized. Moving on to the next category") %(line)
            continue
        
        print("%s is now being resized") %(line)

        os.makedirs(resized_path)
        # iterate over files
        for image_file in IMAGE_FILES:
            try:
                # open file and resize
                im = Image.open(image_file)
                im = im.resize(SIZE, Image.ANTIALIAS)
                #save locally
                output_filename = "%s.jpg" % IMAGE_COUNTER
                im.save(os.path.join('../../resized/', line, output_filename), \
                    "JPEG", quality=70)
                # output
                print("Resizing picture %d") %IMAGE_COUNTER
                # increment image counter
                IMAGE_COUNTER = IMAGE_COUNTER + 1
                # if name == "__main__":
                #     pass
            except:
                print("%s skipped") %output_filename
        os.chdir('..')
