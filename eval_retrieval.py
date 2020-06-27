import json

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
from models.vsepp_model import VSE
from eval_vsepp import encode_data_generated, i2t, t2i_generated

def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def evaluate(json_file):
    print(json_file)
    data = json.load(open(json_file))

    ## evaluate conventional metrics
    # import sys
    # sys.path.append("coco-caption")
    # annFile = '../imgcap/annotations/captions_val2014.json'
    # from pycocotools.coco import COCO
    # from pycocoevalcap.eval import COCOEvalCap
    # coco = COCO(annFile)
    # cocoRes = coco.loadRes(json_file)
    # cocoEval = COCOEvalCap(coco, cocoRes)
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # cocoEval.evaluate()


    print(json_file, len(data))
    imgids = []
    imgs = []
    captions = torch.zeros(len(data), 18).long()
    transform = get_transform('COCO', 'val', None)

    vocab = json.load(open('coco_word2idx.json')) # vocab of vse++ model

    for i, item in enumerate(data):
        imgids.append(item['image_id'])

        # You should change the img_path to the path of COCO validation (test) image
        img_path = '../imgcap/data/raw_images/val2014/COCO_val2014_' + str(item['image_id']).zfill(12) + '.jpg'
        if i % 100 == 0:
            print('load %d images' % i)
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        imgs.append(image.unsqueeze(0))

        cap = item['caption'].split()
        captions[i, 0] = int(vocab['<start>'])

        for j, w in enumerate(cap):
            if j + 1 <= 17:
                captions[i, j+1] = int(vocab[w]) if w in vocab else int(vocab['<unk>'])
            else:
                break
        captions[i, j+2] = int(vocab['<end>'])

    imgs = torch.cat(imgs, 0).contiguous()
    lengths = torch.sum((captions > 0), 1)
    lengths = lengths.cpu().long()

    sorted_lengths, indices = torch.sort(lengths, descending=True)
    new_captions = captions[indices]
    imgs = imgs[indices]
    imgs = imgs.cpu()
    new_captions = new_captions.cpu()

    # You should change the model path to your retrieval model
    model_path = "coco_vse++_resnet_restval_finetune/model_best.pth.tar"
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    model = VSE(opt)
    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Computing results...')
    with torch.no_grad():
        img_embs, cap_embs = encode_data_generated(model, imgs, new_captions, sorted_lengths)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    r, rt = i2t(img_embs, cap_embs, measure=opt.measure, return_ranks=True)
    ri, rti = t2i_generated(img_embs, cap_embs,
                  measure=opt.measure, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)


if __name__ == '__main__':
    evaluate('Tdiv_0.1.json')