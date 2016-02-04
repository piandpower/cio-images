import os

from cimg.transform.transform_image import img_transform

dirname = os.path.dirname
base_path = dirname(dirname(__file__))

src_image_dir = os.path.join(base_path,'data/images')
transformed_image_dir = os.path.join(base_path,'data/_output_transformed_images')

if not os.path.exists(transformed_image_dir):
    os.makedirs(transformed_image_dir)

for img in os.listdir(src_image_dir):
    file, ext = os.path.splitext(img)

    if True: #mimetypes.guess_type(img)[0] is not None and mimetypes.guess_type(img)[0].startswith('image'):
        img_transform(os.path.join(src_image_dir,img),
                      transformed_image_dir)
