import os
import random
import time

from urllib.request import urlopen, urlretrieve
from bs4 import BeautifulSoup




#html = urlopen('http://www.ebay.com/sch/Building-Toys-/18991/i.html?_from=R40&LH_ItemCondition=4&_ipg=192&_nkw=toy&rt=nc&_dmd=1')
html = urlopen('http://www.ebay.com/sch/i.html?_from=R40&_sacat=0&_nkw=tennis%20racquet&_dcat=20871&rt=nc&LH_ItemCondition=3000&_trksid=p2045573.m1684')
bsObj = BeautifulSoup(html.read(),"html.parser")

#print(bsObj.findAll("div",{"id":"ResultSetItems"}).findAll("id",{"r":"3"}).h3)

#print(bsObj.findAll("li",{"r":"3"})[0].h3.a["href"])

save_dir = '/Users/ryoungblood/cio-images/cimg/tests/data/images/ebay_racquet_images'

def findMaxImageUrl(html):
    max_image_loc_start = html.find("maxImageUrl")
    max_image_loc_start += 14
    max_image_loc_stop = html.find('"',max_image_loc_start)
    print(max_image_loc_start)
    img_url = html[max_image_loc_start:max_image_loc_stop]
    img_url= img_url.replace("\\\\u002F","/")
    return img_url

for i in range(53,201):
    print(i)
    rand_time = random.randint(1,7)
    print('rand wait: ',rand_time)
    time.sleep(rand_time)
    product_page = bsObj.findAll("li",{"r":str(i)})[0].h3.a["href"]
    pp_html = urlopen(product_page)
    pp_html_raw = str(pp_html.read())

    img_url = findMaxImageUrl(pp_html_raw)
    new_image_name = "ebay_"+str(i)+os.path.splitext(img_url)[-1]
    print(new_image_name)
    urlretrieve(img_url,os.path.join(save_dir,new_image_name))

    pp_bsObj = BeautifulSoup(pp_html_raw,"html.parser")
    #enlarged_image = pp_bsObj.findAll("img",{"id":"viEnlargeImgLayer_img_ctr"})[0]
    enlarged_image = pp_bsObj.findAll("div",{"id":"mainImgHldr"})[0]#.findAll("img",{"id":"icImg"})[0]
    #enlarged_image = pp_bsObj.findAll("img",{"id":"icImg"})[0]
    if enlarged_image is not None:
        print(enlarged_image)
        #print(enlarged_image["src"])
    else:
        print('no enlarged image')
    print(pp_bsObj.findAll("maxImageUrl"))

    print('-'*50)
    print('waiting')

