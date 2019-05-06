import cv2
import os
import glob

image_folder = 'visualize_loss'
# images = glob.glob(image_folder+"/Iter*.png")
# images.sort(key = os.path.getctime)
# print(images[0])
# frame = cv2.imread(images[0])
# video_name = image_folder+'_revkldiv.avi'
# height, width, layers = frame.shape
# video = cv2.VideoWriter(video_name, 0, 60, (width,height))
# for image in images:
#     video.write(cv2.imread(image))

# cv2.destroyAllWindows()
# video.release()


def create_gif(inputPath, outputPath, delay, finalDelay, loop):
    # grab all image paths in the input directory
    imagePaths = list(glob.glob(inputPath))
    imagePaths.sort(key = os.path.getctime)
    print(imagePaths)
    # remove the last image path in the list
    lastPath = imagePaths[-1]
    imagePaths = imagePaths[:-1]

    # construct the image magick 'convert' command that will be used
    # generate our output GIF, giving a larger delay to the final
    # frame (if so desired)
    cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(
        delay, " ".join(imagePaths), finalDelay, lastPath, loop,
        outputPath)
    os.system(cmd)
    
    
create_gif(image_folder+"/*.png",image_folder+"/output_loss_viz.gif",0.3,200,0)

