import os

from cimg.filter.background_removal import remove_bg

dirname = os.path.dirname
base_path = dirname(dirname(__file__))

src_image_path = os.path.join(base_path,'data/images/boot.jpg')
bgremoved_image_path = os.path.join(base_path,'data/_output_bg-removed_images/boot_bg-removed.jpg')

if not os.path.exists(dirname(bgremoved_image_path)):
    os.makedirs(dirname(bgremoved_image_path))

remove_bg(src_image_path,bgremoved_image_path)

