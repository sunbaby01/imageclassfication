import torch
import torchvision 
import sys
from cnn_finetune import make_model


class ImageClassfication:
    def __init__(self,device=None):
        #device
        if not device:
            self.device=torch.device( "cuda"if torch.cuda.is_available()else "cpu")
        else:
            self.device=torch.device( device)
            
    def load_dataset(self,dataroot="data/train/",batch_size=20):
        """load dataset return dataloder"""
        if not dataroot:dataroot="data/train/" 
        #transforms
        transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop((224,224)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                ])

        self.dataset=torchvision.datasets.ImageFolder(root=dataroot,transform=transform)
        


        self.num_classes=len(self.dataset.classes)
        self.dataset_size=len(self.dataset)
        self.eachclass_size=[0]*self.num_classes
        for i in self.dataset.targets:
            self.eachclass_size[i]+=1     
            
        print("dataset's len =",len(self.dataset))    
        print("eachclass_size =",self.eachclass_size)
        
        dataloader=torch.utils.data.DataLoader(self.dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=4)
        self.dataloader=dataloader
      
        return dataloader

    def load_model(self,modelname="resnet50",pretrained=True):
        """load model return model"""
        if not modelname:modelname="resnet50" 
        model=make_model(modelname, num_classes=self.num_classes, pretrained=pretrained)
        #set model mode
        model.train()

        
        model=model.to(self.device)        
 
        if "cuda" in str(self.device) :
            model=torch.nn.DataParallel(model)
        self.model=model
        print(modelname)
        return model
         
    def save_model(self,model,path="model.pth"):
        """save model"""
        torch.save(model,path)
       
    def train_model(self,epoch=100,lr=0.0005):
        """train model"""
        self.opt=torch.optim.SGD(self.model.parameters(),lr=lr)
        mulsum=1
        #get classes_weight
        classes_weight=[0]*self.num_classes
        for i in self.eachclass_size:
            mulsum*=i
        for i in range(self.num_classes):
            classes_weight[i]= mulsum/ self.eachclass_size[i] 
        classes_weight=torch.tensor(classes_weight,device=self.device)
        classes_weight=classes_weight/torch.sum(classes_weight)
        
        
        self.lossfunction=torch.nn.CrossEntropyLoss( classes_weight )
        print(classes_weight)
        for i in range(epoch):
            losssum=0.0
            accsum=0.0
            recallaccsum=[0.0]*self.num_classes
 
            print("\nepoch",i,":")
            for input,label in self.dataloader:
                input=input.to(self.device)
                label=label.to(self.device)
                
                self.opt.zero_grad()                
                result=self.model(input)                
                loss=self.lossfunction(result,label)
                loss.backward()
                
                result = result.argmax(dim=1)
                losssum+=loss.item()*input.size(0)
                accsum+=torch.sum(label==result)
                for curclass in range(self.num_classes):
                    recallaccsum[curclass]+=torch.sum(torch.tensor([1 if i==j and i == curclass else 0 for i,j in zip(result,label)],device=self.device))
 
                self.opt.step()
                
            self.save_model(self.model,"model.pth")
            
            print("loss:",losssum/self.dataset_size)
            print("acc:",accsum.double().item()/self.dataset_size)
            for j in range(self.num_classes):
                print(j,"acc:",recallaccsum[j].double().item()/self.eachclass_size[j])

def get_finished_model(modelname="resnet50",dataroot="data/train/",batch_size=20):
    m=ImageClassfication()
    m.load_dataset(dataroot=dataroot,batch_size=batch_size)
    m.load_model(modelname)
    m.train_model()
    return m.model
    
if __name__=="__main__":
    dataroot=input("请输入地址or直接回车默认<./data/train>:")
    
    modelname=input("请输入模型名or直接回车默认<resnet50>:")
    
    model=get_finished_model(modelname=modelname,dataroot=dataroot,batch_size=20)
    