import argparse
import cv2
try:
    from fashionStyleClassification import FashionStyleClassification
except ModuleNotFoundError:
    import os, sys
    sys.path.insert(0,os.path.join(os.path.dirname(__file__), os.path.pardir))
    from fashionStyleClassification import FashionStyleClassification

def main(cfg):
    styleClassifier = FashionStyleClassification(cfg.cfg[0])
    
    img = cv2.imread(cfg.img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = styleClassifier.inference(img)
    print(f"style: {res['style']}, score:{res['score']:.4f}")

       
def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Attribute Specific Embedding Network")
    parser.add_argument(
        "--cfg", nargs="+", help="config file", default=None, type=str
    )
    parser.add_argument("--img", default=None, type=str, help="img prefix")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)