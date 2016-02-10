import os

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

both_list = ['brisk','orb','sift','surf']#,'akaze','kaze']
det_list = ['fast','agast','gftt','mser','star','akaze','kaze'] # akaze and kaze should be in the both list, but appears their descriptor functionality only works when they were also the detector
desc_list = ['freak','latch','lucid']
match_list = ['akaze','kaze']

def findAllCombs(a_list,b_list,ab_list,cc_list):
    combs = []
    for a in a_list:
        for b in b_list:
            combs.append((a,b))
        for ab in ab_list:
            combs.append((a,ab))
    for b in b_list:
        for ab in ab_list:
            combs.append((ab,b))
    for ab in ab_list:
        combs.append((ab,ab))
    for cc in cc_list:
        combs.append((cc,cc))
    return combs

# detectors are for finding keypoints.  They often also support computing
detector_descriptor_algs = {
    "akaze": cv2.AKAZE_create(),  # strange delocalization
     "brisk": cv2.BRISK_create(),
     "kaze": cv2.KAZE_create(),  # strange delocalization
     "orb": cv2.ORB_create(),
     "sift": cv2.xfeatures2d.SIFT_create(),
     "surf": cv2.xfeatures2d.SURF_create(),
}

detector_algs = {
    # detectors ONLY
    "agast": cv2.AgastFeatureDetector_create(),
     "fast": cv2.FastFeatureDetector_create(), # lots of points (2159) all over
     "gftt": cv2.GFTTDetector_create(),
     "mser": cv2.MSER_create(),  # very few keypoints (80)
     "star": cv2.xfeatures2d.StarDetector_create(),
     # detectors AND descriptors
     "akaze": detector_descriptor_algs["akaze"],
     "brisk": detector_descriptor_algs["brisk"],
     "kaze": detector_descriptor_algs["kaze"],
     "orb": detector_descriptor_algs["orb"],
     "sift": detector_descriptor_algs["sift"],
     "surf": detector_descriptor_algs["surf"],
    }
descriptor_algs = {
    "freak": cv2.xfeatures2d.FREAK_create(),
    "latch": cv2.xfeatures2d.LATCH_create(),
    "lucid": cv2.xfeatures2d.LUCID_create(1, 1),
    # detectors AND descriptors
     "akaze": detector_descriptor_algs["akaze"],
     "brisk": detector_descriptor_algs["brisk"],
     "kaze": detector_descriptor_algs["kaze"],
     "orb": detector_descriptor_algs["orb"],
     "sift": detector_descriptor_algs["sift"],
     "surf": detector_descriptor_algs["surf"],
}

def compute_features(image, detector_alg, descriptor_alg):
    data = image["image"]
    # if descriptor_alg in detectors:
    #     kps, feature_vectors = detectors[descriptor_alg].detectAndCompute(data, None)
    if detector_alg in detector_algs and descriptor_alg in descriptor_algs:
        keypoints = detector_algs[detector_alg].detect(data)
        keypoints, descriptors = descriptor_algs[descriptor_alg].compute(data, keypoints)
    elif detector_alg not in detector_algs:
        raise ValueError("unknown algorithm passed to detector stage")
    else: # descriptor_alg not in descriptors
        raise ValueError("unknown algorithm passed to descriptor stage")
    image["keypoints"] = keypoints
    image["descriptors"] = descriptors
    return image

def perspective_match(reference, unknown, use_flann=False, min_match_count=10,
                      descriptor=None,nn_dist_ratio_threshold=0.7):
    if use_flann:
        FLANN_INDEX_KDTREE = 0
        FLANN_INDEX_LSH    = 6
        # floating point algorithms
        if descriptor in ["sift", "surf"]:
            index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                                trees = 5)
        # binary algorithms
        else:
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
        search_params = dict(checks = 50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(reference["descriptors"],
                               unknown["descriptors"],
                               k=2)
    good = []
    matchesMask=None
    inferred_homography = None
    for m,n in matches:
        if m.distance < nn_dist_ratio_threshold*n.distance:
            good.append(m)
    if len(good)>min_match_count:
        src_pts = np.float32([ reference["keypoints"][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ unknown["keypoints"][m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # this limits matches to being within the identified subimage
        try:
            inferred_homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
        except AttributeError:
            matchesMask, good, inferred_homography = None, None, None

    else:
        print "Not enough matches are found (%d/%d)" % (len(good), min_match_count)
        matchesMask, good, inferred_homography = None, None, None
    return matchesMask, good, inferred_homography

def draw_matches(reference_features, unknown_features, mask, good_pts):
    fig = plt.figure()
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = (255,0,0),
                       matchesMask = mask,
                       flags = 2)

    img3 = cv2.drawMatches(reference_features["image"],
                           reference_features["keypoints"],
                           unknown_features["image"],
                           unknown_features["keypoints"],
                           good_pts,None,**draw_params)
    plt.imshow(img3)
    return fig

def wrapper(reference, unknown, detector_alg,
            descriptor_alg=None, use_flann=False,
            min_match_count=5,nn_dist_ratio_threshold=0.7,known_homography=None,
            h_match_rtol=0.2,h_match_atol=0.002):
    if not descriptor_alg:
        descriptor_alg = detector_alg
    reference_features = compute_features(reference,
                                          detector_alg,
                                          descriptor_alg)
    unknown_features = compute_features(unknown,
                                        detector_alg,
                                        descriptor_alg)
    print('ref features: ',len(reference_features['keypoints']))
    print('unknown features: ',len(unknown_features['keypoints']))
    matchesMask, good_pts, inferred_homography = perspective_match(reference_features,
                                              unknown_features,
                                             use_flann=use_flann,
                                             min_match_count=min_match_count,
                                             descriptor=descriptor_alg,
                                              nn_dist_ratio_threshold=nn_dist_ratio_threshold)
    h_match = False
    if known_homography is not None and inferred_homography is not None:
        h_match = np.allclose(known_homography,inferred_homography,rtol=h_match_rtol,
                                 atol=h_match_atol)

    fig = draw_matches(reference_features, unknown_features,
                 matchesMask, good_pts)
    fig.gca().set_title("detector: {}, descriptor: {}, Matcher: {}".format(
        detector_alg, descriptor_alg,
        "FLANN" if use_flann else "Brute Force"))

    if good_pts is not None and matchesMask is not None:
        return len(reference_features['keypoints']),len(unknown_features['keypoints']),len(good_pts),sum(matchesMask), h_match, fig
    else:
        return len(reference_features['keypoints']),len(unknown_features['keypoints']),0,0, h_match, fig



def compareDetectorsDescriptors(image_file1,image_file2,results_file=None,
                                save_results_images_dir=None,
                                nn_dist_ratio_threshold=0.7,
                                known_homography=None,
                                h_match_rtol=0.2,h_match_atol=0.002):
    images = {
        "image1": {"filename":image_file1},
        "image2": {"filename":image_file2}
    }

    for image in images:
        images[image]["image"] = cv2.cvtColor(cv2.imread(images[image]["filename"]), cv2.COLOR_BGR2RGB)

    df = pd.DataFrame(findAllCombs(det_list,desc_list,both_list,match_list),#[('surf','surf'),('kaze','kaze')],#[('agast','sift'),('agast','latch')],
                  columns=['detector','descriptor'])
    df['combo'] = df['detector']+df['descriptor']
    df = df.set_index('combo')

    for row in df.index:
        print(row)
        print(df.loc[row,'detector'])
        print(df.loc[row,'descriptor'])
        df.loc[row,'img1_kps'],df.loc[row,'img2_kps'],df.loc[row,'num_matches'], df.loc[row,'num_h_matches'], df.loc[row,'h_match'], fig = wrapper(images["image1"],
                                            images["image2"],
                                            detector_alg=df.loc[row,'detector'],
                                            descriptor_alg=df.loc[row,'descriptor'],
                                            nn_dist_ratio_threshold=nn_dist_ratio_threshold,
                                            known_homography=known_homography,
                                            h_match_rtol=h_match_rtol,
                                            h_match_atol=h_match_atol)#,
                                            #use_flann=True)
        pct_match = df.loc[row,'num_h_matches']/min(df.loc[row,'img1_kps'],df.loc[row,'img2_kps'])
        df.loc[row,'pct_match'] = pct_match
        df.loc[row,'pct_inliers'] = df.loc[row,'num_h_matches'] / df.loc[row,'num_matches']
        print(df.loc[row,'num_h_matches'])
        print('-'*40)

        if save_results_images_dir is not None and os.path.exists(save_results_images_dir) and df.loc[row,'num_matches'] != 0:
            combo = df.loc[row,'detector']+'_'+df.loc[row,'descriptor']
            plot_path = os.path.join(save_results_images_dir,os.path.splitext(os.path.basename(results_file))[0]+'_'+combo+'.jpg')
            plt.savefig(plot_path)

    if results_file is not None:
        df.sort_values(['h_match','num_h_matches','pct_inliers'], ascending=False).to_csv(results_file)

