import os
import numpy as np

from cimg.feature.compare_detectors_descriptors import compareDetectorsDescriptors

dirname = os.path.dirname
base_path = dirname(dirname(__file__))

src_image_dir    = os.path.join(base_path,'data/images/mikolajczyk_benchmark/bark')
results_dir      = os.path.join(base_path,'data/_output_comparison_results/bark')

image_pairs = [
    #(os.path.join(src_image_dir,'01.jpg'),os.path.join(src_image_dir,'05.png'))
    (os.path.join(src_image_dir,'img1.ppm'),os.path.join(src_image_dir,'img4.ppm'),os.path.join(src_image_dir,'H1to4p'))
]

for image_pair in image_pairs:
    image1_name = os.path.splitext(os.path.basename(image_pair[0]))[0]
    image2_name = os.path.splitext(os.path.basename(image_pair[1]))[0]
    known_homography = np.genfromtxt(image_pair[2])

    results_imgs_dir = os.path.join(results_dir,'images_'+image1_name+'_vs_'+image2_name)
    if not os.path.exists(results_imgs_dir):
        os.makedirs(results_imgs_dir)

    results_file = os.path.join(results_dir,image1_name+'_vs_'+image2_name+'.csv')
    compareDetectorsDescriptors(image_pair[0],image_pair[1],
                                results_file=results_file,
                                save_results_images_dir=results_imgs_dir,
                                nn_dist_ratio_threshold=0.75,
                                known_homography=known_homography,
                                h_match_rtol=0.2,h_match_atol=0.002)