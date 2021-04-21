import numpy as np 
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms as T
import torchvision
import segmentation_models_pytorch as smp
#weight file hosted at https://f000.backblazeb2.com/file/fypLanding/Unet-Mobilenet.pt

class SegmentationEngine():
    def __init__(self, pathWeights):
        self.net=torch.load(pathWeights)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def inferImage(self,img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.net.eval()
        t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        image = t(img)
        self.net.to(self.device); image=image.to(self.device)
        with torch.no_grad():
            image = image.unsqueeze(0)        
            output = self.net(image)
            mask = torch.argmax(output, dim=1)
            mask = mask.cpu().squeeze(0)
        return mask

    def maskToPIL(self,mask):
        maskNp=mask.numpy()
        img = Image.fromarray(np.uint8(maskNp) , 'L')   
        return img

if __name__ == '__main__':

    import glob
    IMAGE_DIR = '/content/Safe-UAV-Landing/data/test/images/*.jpg'
    pathWeights='/content/Safe-UAV-Landing/models/yolo-v3/Unet-Mobilenet.pt'
    seq=glob.glob(IMAGE_DIR)
    segEngine=SegmentationEngine(pathWeights)
    
    for file in seq:
        i=3
        img = Image.open(file)
        file_name=Path(file).stem
        img = img.resize((1152, 768), Image.ANTIALIAS)
        pred_mask=segEngine.inferImage(img)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Picture');
        ax2.imshow(pred_mask)
        ax2.set_title('Prediction')
        ax2.set_axis_off()
        pred_mask.numpy()
        xxx=pred_mask.numpy()
        img = Image.fromarray(np.uint8(xxx) , 'L')
        print('success')