import pdb
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F
import cv2


def save_img(image, img_name):
    image = image.permute(1,2,0)
    image = (128*image.detach().cpu().numpy()) + 127.5
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_name, image)



class VeriFace(object):
    def __init__(self, threshold=0.6):
        self.THRESHOLD = threshold
        self.device = torch.device('cuda')
        self.mtcnn = MTCNN(image_size=224,
                  margin=60,
                  device=self.device).eval()

        # Create an inception resnet (in eval mode):
        self.resnet = InceptionResnetV1(pretrained='vggface2',
                               device=self.device,
                               classify=False).eval()
    
    def verify(self, image, doc_image):

        dets_id = self.mtcnn(image)
        dets_doc = self.mtcnn(doc_image)

        encoding_image = self.resnet(dets_id.unsqueeze(0).to(self.device))
        encoding_doc = self.resnet(dets_doc.unsqueeze(0).to(self.device))
        score = F.cosine_similarity(encoding_image, encoding_doc)
        if score > self.THRESHOLD:
            return round(score.item(), 2), True
        else:
            return round(score.item(), 2), False



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_path', type=str, help='path to document')
    parser.add_argument('--face_path', type=str, help='path to person\'s image')
    args = parser.parse_args()
    image = Image.open(args.face_path)
    doc_image = Image.open(args.doc_path)
    verifier = VeriFace()
    score, match = verifier.verify(image, doc_image)

    result = 'Match' if match else 'Not a Match'
    result = f'{result} with score:{score}'

    fig = plt.figure()
    fig.suptitle(result)
    
    fig.add_subplot(1,2,1)
    plt.imshow(image)
    plt.title("Person")

    fig.add_subplot(1,2,2)
    plt.imshow(doc_image)
    plt.title("Document")

    fig.savefig('demo_out5.jpg')






