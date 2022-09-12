import os.path as osp
import shutil

from PIL import Image
from PIL import ImageDraw

base_dir = '/Users/oddconcetps/dfd_gan_results'


def image_crop(img_path, chopsize=256):

    if isinstance(img_path, str):
        img = Image.open(img_path).convert('RGB')
    else:
        img = img_path
    width, height = img.size
    return_list = list()
    for x0 in range(0, width, chopsize):
        for y0 in range(0, height, chopsize):
            box = (x0, y0,
                   x0 + chopsize if x0 + chopsize < width else width,
                   y0 + chopsize if y0 + chopsize < height else height)
            crop_img = img.crop(box)
            return_list.append(crop_img)

    return return_list

# https://note.nkmk.me/en/python-pillow-concat-images/
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def draw_figure(outfit_id, idx, result_list, n_result=3, img_size=256):
    #black_image = Image.new('RGB', (img_size, img_size))
    dfd_result = result_list[0][0]
    gt_result = result_list[0][1]
    #result_img = get_concat_h(black_image, dfd_result)
    result_img = get_concat_h(gt_result, dfd_result)
    pix_result = result_list[1]
    result_img = get_concat_h(result_img, pix_result)
    cyc_result = result_list[2]
    result_img = get_concat_h(result_img, cyc_result)
    source_list = result_list[0][2:]
    source_img = source_list[0]
    for i in range(1, len(source_list)):
        source_img = get_concat_h(source_img, source_list[i])

    result_img = get_concat_h(source_img, result_img)
    result_img.save(osp.join('figures', 'compare_fig', f'{outfit_id}_{idx}_combine.jpg'))
    result_img.close()


def get_figure_compare():

    if len(target_dir_list) == 0 and len(outfit_id_list) == 0:
        print(f'Please check target_dir_list and outfit_id_list')
        return

    for outfit_id, idx in outfit_id_list:

        img_list, result_list = list(), list()
        for target_dir in target_dir_list:
            check_img = osp.join(base_dir, target_dir, f'{outfit_id}_{idx}.jpg')
            assert osp.exists(check_img), check_img
            img_list.append(check_img)

        for index, img_path in enumerate(img_list):
            if index == 0:
                result_list.append(image_crop(img_path, chopsize=256))
            else:
                result_list.append(image_crop(img_path, chopsize=256)[0])
        draw_figure(outfit_id, idx, result_list)


def concat_v_fig(img_list, target_dir, fname, fisrt_copy=True):
    img_list = [osp.join(target_dir, x) for x in img_list]

    if fisrt_copy:
        shutil.copy(img_list[0], fname.replace('.jpg', '_single.jpg'))
    img_list = [Image.open(x).convert('RGB') for x in img_list]
    pading = Image.new('RGB', (256 * len(img_list), 5))

    result = img_list[0]

    for i in range(1, len(img_list)):
        result = get_concat_v(result, pading)
        result = get_concat_v(result, img_list[i])
    result.save(fname)


def make_tsne_fig():
    base_dir = 'figures'
    imgs = ['map_train_p100_tsne.png', 'no_map_train_p100_tsne.png']
    imgs = [osp.join(base_dir, x) for x in imgs]
    imgs = [Image.open(x).convert('RGB') for x in imgs]

    result = get_concat_h(imgs[0], imgs[1])
    result.save(f'figures/tsne_result_new.jpg')


def make_wrong_fig():
    base_dir = 'figures/wrong_case'
    img_list1 = ['11830812_5.jpg', '73201386_1.jpg', '59866572_3.jpg', '153031853_3.jpg']
    img_list1 = [osp.join(base_dir, x) for x in img_list1]
    img_list1 = [Image.open(x).convert('RGB') for x in img_list1]
    img_list2 = ['11830812_5.jpg', '59866572_3.jpg']
    img_list2 = [osp.join(base_dir, x) for x in img_list2]
    img_list2 = [Image.open(x).convert('RGB') for x in img_list2]

    def func(img_list, name='example'):
        result = img_list[0]
        for i in range(1, len(img_list)):
            result = get_concat_v(result, img_list[i])
        result.save(f'figures/wrong_case/{name}')

    func(img_list1, 'wrong_case1.jpg')
    func(img_list2, 'wrong_case2.jpg')


def get_fig_gen_single():
    def get_text_gen():
        from PIL import Image
        from PIL import ImageFont
        from PIL import ImageDraw

        font1 = ImageFont.truetype("times-ro.ttf", 50)
        font2 = ImageFont.truetype("times-ro.ttf", 45)

        img1 = Image.new('RGB', (256 * 4, 80), (255, 255, 255))
        img2 = Image.new('RGB', (256 * 4, 80), (255, 255, 255))

        draw1 = ImageDraw.Draw(img1)
        draw2 = ImageDraw.Draw(img2)
        for i in range(4):
            draw1.text((256 * i + 50, 40), f"Source {i + 1}", (0, 0, 0), font=font1)

        draw2.text((256 * 0 + 20, 40), "Ground truth", (0, 0, 0), font=font2)
        draw2.text((256 * 1 + 45, 40), "DFDGAN", (0, 0, 0), font=font2)
        draw2.text((256 * 2 + 60, 40), "pix2pix", (0, 0, 0), font=font2)
        draw2.text((256 * 3 + 30, 40), "CycleGAN", (0, 0, 0), font=font2)
        return img1, img2

    base_dir_fig = 'figures/fig_generation'
    img_dict = {
        'earrings': ['74582924_1_combine.jpg'], # '190887254_4_combine.jpg', '53911173_2_combine.jpg'
        'bags': ['74582924_5_combine.jpg'], # '16189792_5_combine.jpg', '59926547_5_combine.jpg',
        'bottoms':['57573339_3_combine.jpg'], # '54875268_3_combine.jpg', '51569972_2_combine.jpg'
        'eye_glasses':['216427095_5_combine.jpg'], # '32343336_5_combine.jpg', '9237503_5_combine.jpg'
        'inner':['11241532_1_combine.jpg'], # '99557450_1_combine.jpg', '180421125_1_combine.jpg'
        'outer':['26975778_2_combine.jpg'], # '51315322_2_combine.jpg', '54099336_4_combine.jpg'
        'shoes':['45795275_4_combine.jpg'], # '9429822_4_combine.jpg', '81384001_4_combine.jpg'
    }

    for cate in img_dict:

        img_list = img_dict[cate]

        for img_path in img_list:
            #print(Image.open(osp.join(base_dir_fig, cate, img_path)).convert('RGB').size)
            r_list = image_crop(osp.join(base_dir_fig, cate, img_path))
            #for im in r_list:
            #    print(im.size)
            s1,s2,s3,s4,gt,dfd,pix,cyc = r_list

            source_list = get_concat_h(get_concat_h(get_concat_h(s1,s2), s3), s4)
            comp_list = get_concat_h(get_concat_h(get_concat_h(gt,dfd), pix), cyc)
            s_t, c_t = get_text_gen()
            result_img = get_concat_v(get_concat_v(s_t, source_list), get_concat_v(c_t, comp_list))
            result_img.save(osp.join(base_dir_fig, f'{cate}_fig_gen_single.jpg'))


def get_fig_gen():
    def get_text_gen():
        from PIL import Image
        from PIL import ImageFont
        from PIL import ImageDraw

        font = ImageFont.truetype("times-ro.ttf", 34)

        img = Image.new('RGB', (256 * 8, 50), (255, 255, 255))

        draw = ImageDraw.Draw(img)
        for i in range(4):
            draw.text((256 * i + 65, 25), f"Source {i + 1}", (0, 0, 0), font=font)
        draw.text((1024 + 256 * 0 + 32, 25), "Ground truth", (0, 0, 0), font=font)
        draw.text((1024 + 256 * 1 + 48, 25), "DFDGAN", (0, 0, 0), font=font)
        draw.text((1024 + 256 * 2 + 64, 25), "pix2pix", (0, 0, 0), font=font)
        draw.text((1024 + 256 * 3 + 32, 25), "CycleGAN", (0, 0, 0), font=font)

        # img.save('compare_title.jpg')
        return img

    base_dir_fig = 'figures/fig_generation'
    img_dict = {
        'earrings': ['74582924_1_combine.jpg', '190887254_4_combine.jpg', '53911173_2_combine.jpg'], #
        'bags': ['59866572_5_combine.jpg', '59926547_5_combine.jpg', '74582924_5_combine.jpg'], #
        'bottoms':['54875268_3_combine.jpg', '57573339_3_combine.jpg', '51569972_2_combine.jpg'], #
        'eye_glasses':['32343336_5_combine.jpg', '216427095_5_combine.jpg', '9237503_5_combine.jpg'], #
        'inner':['11241532_1_combine.jpg', '99557450_1_combine.jpg', '180421125_1_combine.jpg'], #
        'outer':['26975778_2_combine.jpg', '51315322_2_combine.jpg', '57573339_2_combine.jpg'], #
        'shoes':['9429822_4_combine.jpg', '45795275_4_combine.jpg', '81384001_4_combine.jpg'], #
    }

    for cate in img_dict:

        img_list = img_dict[cate]
        pil_imgs = [Image.open(osp.join(base_dir_fig, cate, x)).convert('RGB') for x in img_list]

        result_img = pil_imgs[0]

        for i in range(1, len(pil_imgs)):
            result_img = get_concat_v(result_img, pil_imgs[i])
        get_concat_v(get_text_gen(), result_img).save(osp.join(base_dir_fig, f'{cate}_fig_gen.jpg'))


def get_fig_refine():

    def get_text_refine(len):
        from PIL import Image
        from PIL import ImageFont
        from PIL import ImageDraw

        font2 = ImageFont.truetype("times-ro.ttf", 45)

        img = Image.new('RGB', (256 * len, 80), (255, 255, 255))

        draw = ImageDraw.Draw(img)
        for i in range(len):
            if i % 2 == 0:
                draw.text((0 + 256 * i + 20, 40), "Generation", (0, 0, 0), font=font2)
            else:
                draw.text((0 + 256 * (i) + 20, 40), "Refinement", (0, 0, 0), font=font2)

        return img

    base_dir_refine = 'figures/fig_refinement'
    img_list = ['9429822_4.jpg', '10525879_1.jpg',
                '7238896_5.jpg', '16189792_5.jpg',
                '153031853_3.jpg', '31167398_5.jpg',] #

    box_list = [[(60,120,124,184),],
                [(128,128,182,182),],
                [(60,20,124,84),],
                [(110,110,174,174),],
                [(90,45,154,114),],
                [(120,5,184,69),]]

    box_list = box_list[:len(img_list)]

    assert len(img_list) == len(box_list), 'Check len of img_list and box_list'

    result_list = list()

    for img_path, bb_list in zip(img_list, box_list):
        gt, refine, _, _, _, _ = image_crop(osp.join(base_dir_refine, img_path), chopsize=256)

        draw_gt = ImageDraw.Draw(gt)
        draw_refine = ImageDraw.Draw(refine)

        gt_crops, ref_crops = list(), list()

        for idx, bbox in enumerate(bb_list):

            color = (255, 0, 0) if idx == 0 else (0, 255, 0)

            draw_gt.rectangle(bbox, outline=color, width=1)
            draw_refine.rectangle(bbox, outline=color, width=1)

            gt_crop_b = gt.crop(bbox)
            gt_crops.append(gt_crop_b)
            ref_crop_b = refine.crop(bbox)
            ref_crops.append(ref_crop_b)

        gt_final = get_concat_v(gt, gt_crops[0].resize((256, 128)))
        ref_final = get_concat_v(refine, ref_crops[0].resize((256, 128)))

        temp_img = get_concat_h(gt_final, ref_final)
        result_list.append(temp_img)

    def concat_h_list(input_list):
        result_img = input_list[0]
        for i in range(1, len(input_list)):
            result_img = get_concat_h(result_img, input_list[i])
        return result_img

    half_cut = len(result_list) // 2

    top_img = concat_h_list(result_list[:half_cut])
    bottom_img = concat_h_list(result_list[half_cut:])

    #result_img.save(osp.join(base_dir_refine, 'refine_fig.jpg'))
    get_concat_v(get_text_refine(len(img_list)), get_concat_v(top_img, bottom_img)).save(osp.join(base_dir_refine, f'refine_fig_{len(img_list)}.jpg'))


def get_fig_wrong():

    def get_text_wrong():
        from PIL import Image
        from PIL import ImageFont
        from PIL import ImageDraw

        font1 = ImageFont.truetype("times-ro.ttf", 55)
        font2 = ImageFont.truetype("times-ro.ttf", 50)
        font3 = ImageFont.truetype("times-ro.ttf", 45)

        img = Image.new('RGB', (256 * 6, 80), (255, 255, 255))

        draw = ImageDraw.Draw(img)
        for i in range(4):
            draw.text((256 * i + 33, 40), f"Source {i + 1}", (0, 0, 0), font=font1)
        draw.text((1024 + 256 * 0 + 15, 40), "Generation", (0, 0, 0), font=font2)
        draw.text((1024 + 256 * 1 + 13, 40), "Ground truth", (0, 0, 0), font=font3)

        # img.save('compare_title.jpg')
        return img

    img_list = ['27512972_5.jpg', '11830812_5.jpg', '59866572_3.jpg', '136057135_1.jpg']
    base_dir_wrong = f'figures/fig_wrong'
    result_list = list()

    for img_path in img_list:
        fake, gt, s1, s2, s3, s4 = image_crop(osp.join(base_dir_wrong, img_path), chopsize=256)

        source_item = s1

        for source in [s2, s3, s4]:
            source_item = get_concat_h(source_item, source)

        result_img = get_concat_h(source_item, fake)
        result_img = get_concat_h(result_img, gt)
        result_list.append(result_img)

    result_img = result_list[0]

    for i in range(1, len(result_list)):
        result_img = get_concat_v(result_img, result_list[i])

    get_concat_v(get_text_wrong(), result_img).save(osp.join(base_dir_wrong, 'wrong_fig.jpg'))


def get_fig_outfit_exploration():
    def get_text_outfit_explor():
        from PIL import Image
        from PIL import ImageFont
        from PIL import ImageDraw

        font1 = ImageFont.truetype("times-ro.ttf", 55)
        font2 = ImageFont.truetype("times-ro.ttf", 50)

        img = Image.new('RGB', (256 * 5, 80), (255, 255, 255))

        draw = ImageDraw.Draw(img)
        for i in range(4):
            draw.text((256 * i + 33, 40), f"Source {i + 1}", (0, 0, 0), font=font1)
        draw.text((1024 + 256 * 0 + 10, 40), "Refinement", (0, 0, 0), font=font2)

        # img.save('compare_title.jpg')
        return img

    img_list = ['paper_191678943_5_35932638_1_13.jpg', 'paper_191678943_5_10396291_2_2.jpg', 'paper_191678943_5_31913405_1_20.jpg', # eye_glasses
                'paper_25612204_3_147449622_2_5.jpg', 'paper_25612204_3_32184120_2_1.jpg', 'paper_25612204_3_32256188_2_3.jpg', # shoes
                ]
    base_dir_wrong = f'figures/fig_outfit_exploration'
    result_list = list()

    f_red, s_red = 0, 1

    for img_path in img_list:
        fake, refine, s1, s2, s3, s4 = image_crop(osp.join(base_dir_wrong, img_path), chopsize=256)

        source_item = s1

        for source in [s2, s3, s4]:
            source_item = get_concat_h(source_item, source)

        result_img = get_concat_h(source_item, refine)
        result_list.append(result_img)

    result_img = result_list[0]

    for i in range(1, len(result_list)):
        result_img = get_concat_v(result_img, result_list[i])

    img = get_concat_v(get_text_outfit_explor(), result_img)\
    #.save(osp.join(base_dir_wrong, 'outfit_explor_fig.jpg'))

    draw = ImageDraw.Draw(img)
    draw.rectangle((f_red * 256, 80, 256 * (f_red + 1), 75 + 256 * 3), outline=(255, 0, 0), width=5)
    draw.rectangle((s_red * 256, 70 + 256 * 3, 256 * (s_red + 1), 73 + 256 * 6), outline=(255, 0, 0), width=5)

    img.save(osp.join(base_dir_wrong, 'outfit_explor_fig.jpg'))

def get_fig_tsne():

    m_tsne_fig = Image.open('figures/fig_tsne/mapping_train_p100_dot5_tsne.png').convert('RGB')
    nm_tsne_fig = Image.open('figures/fig_tsne/non_mapping_train_p100_dot5_tsne.png').convert('RGB')

    m_tsne_fig = m_tsne_fig.crop((30, 0, 640, 480))
    m_tsne_fig = m_tsne_fig.crop((0, 0, 580, 480))

    nm_tsne_fig = nm_tsne_fig.crop((30, 0, 640, 480))
    nm_tsne_fig = nm_tsne_fig.crop((0, 0, 580, 480))
    #m_tsne_fig.save(f'figures/fig_tsne/m_tsne_crop.jpg')

    def get_img(img):
        from PIL import Image
        from PIL import ImageFont
        from PIL import ImageDraw

        font1 = ImageFont.truetype("times-ro.ttf", 40)

        #img = Image.new('RGB', (1280, 80), (255, 255, 255))

        draw = ImageDraw.Draw(img)
        draw.text((235, 25), "Mapping", (0, 0, 0), font=font1)
        draw.text((215 + 560, 25), "Non-Mapping", (0, 0, 0), font=font1)
        return img

    tsnes_img = get_concat_h(m_tsne_fig, nm_tsne_fig)
    get_img(tsnes_img).save(osp.join(f'figures/fig_tsne/tsne_result.jpg'))


if __name__ == '__main__':
    outfit_id_list = [[9237503, 5], [11007498, 3], [8401543, 5],
                      [9429822, 4], [11241532, 1], [11241532, 2],
                      [11241532, 5], [13054075, 2], [13054075, 3],
                      [16189792, 5], [17029923, 3], [17029923, 4],
                      [20899970, 3], [20899970, 4], [26975778, 2],
                      [31167398, 4], [31167398, 5], [31314342, 5],
                      [32285957, 4], [32899949, 4], [32899949,5],
                      [37010366, 4], [40068057, 1], [40068057, 2], [40068057, 3],
                      [40068057, 5], [40217407, 4], [41474592, 5], [43803205, 4],
                      [45795275, 4], [7238896, 5], [9237503, 5], [46431754, 2],
                      [46431754, 5], [51315322, 2], [51315322, 3], [54099336, 4],
                      [54099336, 5], [54191540, 2], [54191540, 3], [54875268, 3],
                      [57573339, 2], [57573339, 3], [57573339, 4], [58000076, 2],
                      [58000076, 5], [59004080, 3], [61105690, 3], [71126120, 4],
                      [73121408, 5], [73201386, 1], [73974284, 2], [73974284, 3],
                      [74221095, 3], [74582924, 1], [74582924, 4], [74582924, 5],
                      [76698719, 4], [79363665, 3], [81384001, 4], [86836715, 2],
                      [86836715, 5], [94027475, 2], [96576865, 3], [99557450, 1],
                      [106894554, 4], [110509119, 4], [136057135, 3], [141396794, 5],
                      [143433435, 2], [154301313, 3], [174776889, 3], [180421125, 1],
                      [183828857, 2], [190887254, 4], [197606699, 5], [200348230, 4],
                      [208048721, 3], [209459743, 1], [216067846, 5], [216210379, 5],
                      [216427095, 5], [216493769, 1], [216493769, 3], [216789526, 3],
                      [216807508, 4], [216807508, 3], [216862072, 5], [26975778, 5],
                      [45917394, 5], [53911173, 2], [74582924, 1], [81384001, 4],
                      [208750775, 4], [208773219, 4], [216210379, 5], [216427095, 5],
                      [216753232, 5], [45917394, 5], [51569972, 2], [54875268, 3], [32343336, 5],
                      [10525879, 1], [11976118, 5], [12266317, 3], [13054075, 3], [22079966, 3],
                      [54099336, 4], [59866572, 5], [59926547, 5], [76098746, 2], [94027475, 2],
                      [142176992, 5], [190887254, 4], [211635916, 3], [213570086, 4]
                      ]
    target_dir_list = ['dfd_gan_test_500', 'pix2pix_unet_test_500', 'cycle_resnet_test_500']
    #get_figure_compare()
    #make_wrong_fig()
    #make_tsne_fig()
    #compare_text()

    #compare_img = get_concat_v(compare_text_fig_g(), Image.open('./figures/compare_fig/7238896_5_combine.jpg'))
    #compare_img.save('compare_img.jpg')

    # get generation figure
    #get_fig_gen_single()
    # get generation figure
    #get_fig_gen()
    # get refinement figure
    get_fig_refine()
    # get wrong figure
    #get_fig_wrong()
    # get outfit exploration
    #get_fig_outfit_exploration()

    #result = image_crop(Image.new('RGB', (256 * 10, 80), (255, 255, 255)), chopsize=256)
    #for x in result:
    #    print(x.size)
    #print(get_concat_h(Image.new('RGB', (256, 256), (255, 255, 255)), Image.new('RGB', (256, 256), (255, 255, 255))).size)
    #print(get_concat_v(Image.new('RGB', (256, 256), (255, 255, 255)), Image.new('RGB', (256, 256), (255, 255, 255))).size)

    # get tsne result
    #get_fig_tsne()