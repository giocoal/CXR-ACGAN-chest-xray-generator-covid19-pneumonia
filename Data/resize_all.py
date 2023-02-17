import os
from PIL import Image


def resize_im(path):
    if os.path.isfile(path):
        im = Image.open(path).resize((224,224), Image.ANTIALIAS)
        parent_dir = os.path.dirname(path)
        img_name = os.path.basename(path).split('.')[0]
        os.remove(path)
        im.save(os.path.join(parent_dir, img_name + '.png'), 'PNG', quality=90)

def resize_all(mydir):
    count = 0
    count_failed = 0
    for subdir , _ , fileList in os.walk(mydir):
        for f in fileList:
            print(count)
            count += 1
            try:
                full_path = os.path.join(subdir,f)
                resize_im(full_path)
            except Exception as e:
                print('Unable to resize %s. Skipping.' % full_path)
    print(f"Total unsuccessful: {count_failed}")
                
if __name__ == '__main__':
    resize_all("./COVIDx-splitted-resized-112")
    