import os

from cimg.feature.compare_detectors_descriptors import compareDetectorsDescriptors

dirname = os.path.dirname
base_path = dirname(dirname(__file__))

src_image_dir    = os.path.join(base_path,'data/images')
results_dir      = os.path.join(base_path,'data/_output_comparison_results')
results_imgs_dir = os.path.join(results_dir,'images')
#transformed_image_dir = os.path.join(base_path,'data/_output_transformed_images')

if not os.path.exists(results_imgs_dir):
    os.makedirs(results_imgs_dir)

image_pairs = [
    (os.path.join(src_image_dir,'1.jpg'),os.path.join(src_image_dir,'2.jpg'))
]

for image_pair in image_pairs:
    image1_name = os.path.splitext(os.path.basename(image_pair[0]))[0]
    image2_name = os.path.splitext(os.path.basename(image_pair[1]))[0]

    results_file = os.path.join(results_dir,image1_name+'_vs_'+image2_name+'.csv')
    compareDetectorsDescriptors(image_pair[0],image_pair[1],
                                results_file=results_file,
                                save_results_images_dir=results_imgs_dir)