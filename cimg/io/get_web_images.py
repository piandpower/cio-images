import os
import urllib
import cv2

def get_web_images(url_list_file=None,credentials_file=None,save_dir=None):
    with open(credentials_file,'r') as c:
        creds = c.readline().rstrip() # rstrip strips the newline character
        with open(url_list_file,'r') as f:
            for i, line in enumerate(f):
                line = line.rsplit('\n',1)[0] # strip newline character

                url = line.split('//',1)[0] + '//' + creds +line.split('//',1)[1]
                image = url.rsplit('/',1)[1]

                img_path = os.path.join(save_dir,image)
                urllib.urlretrieve(url,img_path)

                file, ext = os.path.splitext(os.path.basename(img_path))

                if ext is "":
                    img = cv2.imread(img_path)
                    cv2.imwrite(img_path+'.jpg',img)
                    os.remove(img_path)