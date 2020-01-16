import torch
import torchvision 
import sys,os 
from PIL import Image

class TestModel:
    def __init__(self,device=None):
        #device
        if not device:
            self.device=torch.device( "cuda"if torch.cuda.is_available()else "cpu")
        else:
            self.device=torch.device( device)

    def get_imagepath_list(self,dataroot="data/val/"):
        if not dataroot:dataroot="data/val/"
        
        IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]
        
        def is_image_file(filename):
            return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
            
        assert os.path.isdir(dataroot)

        image_paths = []
        for root, _, fnames in sorted(os.walk(dataroot,followlinks=True)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    image_paths.append(path)
        self.image_paths=image_paths            
        return image_paths

    def load_model(self,path="model.pth"):
        model=torch.load(path,map_location=self.device)
        model.eval()
        model=model.to(self.device)
        self.model=model
        return model
         
 
     
    def test_model(self):
             
        transform=torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            ])

        for input in self.image_paths:
            input=Image.open(input).convert('RGB')
            input=transform(input)
            input=input.view(-1,3,224,224)
            input=input.to(self.device) 
            result=self.model(input)
            print(result.argmax(dim=1).item() )

            
            


    
    
    
if __name__=="__main__":
    dataroot=input("请输入地址or直接回车默认<./data/val>:")
    m=TestModel()

    m.get_imagepath_list(dataroot)

    m.load_model()
    m.test_model()
