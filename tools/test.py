import argparse
import os, sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__), os.path.pardir))
import glob
import json
import cv2
from tqdm import tqdm
from fashionStyleClassification import FashionStyleClassification

id2cls = {0:'formal', 1:'semi-formal', 2:'casual'}


def main(cfg):
    jsons=glob.glob(os.path.join(args.json_prefix, '**/*.json'), recursive=True)
    img_prefix = args.img_prefix
    styleClassifier = FashionStyleClassification(cfg.cfg[0])
    for jsonpath in tqdm(jsons):
        
        with open(jsonpath,'r') as f:
            labels = json.load(f)
        merged_label = {'meta_data':labels['meta_data']}
        merged_label['Contents_tag']={'caption_tag':[]}
        merged_label['Caption_data']={'caption_data':''}
        imgpath = os.path.join(img_prefix, os.path.basename(jsonpath)).replace('.json', '.jpg')
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        obj_infos = []
        pose_info = labels['pose_info']
        for i, obj in enumerate(pose_info):
            bbox = obj.pop('bbox')
            tlx,tly, brx, bry = int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[0]))+int(float(bbox[2])),  int(float(bbox[1]))+int(float(bbox[3]))
            crop = img[tly:bry, tlx:brx].copy()
            
            res = styleClassifier.inference(crop)
           
            if args.visualize:
                imgname = os.path.basename(imgpath)
                imgname = imgname.split('.')[0]
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                #cv2.imwrite(f'visualize/{imgname}_{i}_{res['style']}_{res['score']:.4f}.jpg', crop)

            
            obj_infos.append({'bbox':bbox,
            'pose_info':obj,
            'face_info':{
            'sex':'',
            'age':'',
            'num_Landmark':'',
            'bbox':[],
            'Landmark':[],
            'Emotion':''
            },
            'style_info':f'{cls_id.item()}',
            })
        
        merged_label['Contents_info']={'object_contents_info':{'MAIN_CHARAC':'','count':len(obj_infos),'CNTS':obj_infos},
        'brand_info':{'bbox':[], 'TEXT':''},
        'QR_info':{'bbox':[], 'url':''},
        'OCR_info':{'num_word':'', 'words':[{'bbox':[], 'TEXT':'','class':''},]}
        }
        
        jsonname = os.path.basename(jsonpath)
        with open(os.path.join(args.json_outpath,jsonname), 'w') as f:
            json.dump(merged_label, f, indent=2)

       

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Attribute Specific Embedding Network")
    parser.add_argument(
        "--cfg", nargs="+", help="config file", default=None, type=str
    )
    parser.add_argument("--img_prefix", default=None, type=str, help="img prefix")
    parser.add_argument("--json_prefix", default=None, type=str, help="path to directory of input json files")
    parser.add_argument("--json_outpath", default=None, type=str, help="path to directory of output json files")
    parser.add_argument("--visualize", action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # if args.cfg is not None:
    #     for cfg_file in args.cfg:
    #         cfg.merge_from_file(cfg_file)
    
    # cfg.freeze()
    main(args)