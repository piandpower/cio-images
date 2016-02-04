import os

from cimg.io.get_web_images import get_web_images

dirname = os.path.dirname
base_path = dirname(dirname(__file__))

url_list_file = os.path.join(base_path,'data/atf_image_urls.txt')
cred_file = os.path.join(base_path,'data/PRIVATE_CREDENTIALS.txt')
save_image_dir = os.path.join(base_path,'data/_output_downloaded_images/')

if not os.path.exists(save_image_dir):
    os.makedirs(save_image_dir)

get_web_images(url_list_file=url_list_file,credentials_file=cred_file,
               save_dir=save_image_dir)