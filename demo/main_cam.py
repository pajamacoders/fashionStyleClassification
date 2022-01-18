import argparse
import cv2
from fashionStyleClassification import FashionStyleClassification

def main(cfg):
    styleClassifier = FashionStyleClassification(cfg.cfg[0])
    cam =cv2.VideoCapture(0)
    if cam.isOpened():
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #영상 width 설정
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #영상 height 설정
        while True:
            ret, img =cam.read() #영상 읽기
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = styleClassifier.inference(img)
            print(f"style: {res['style']}, score:{res['score']:.4f}")
            cv2.imshow('img', img)
            if cv2.waitKey(1) == 27 : #esc 누르면 종료
            	break
 

       
def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Attribute Specific Embedding Network")
    parser.add_argument(
        "--cfg", nargs="+", help="config file", default=None, type=str
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
