#coding:utf-8
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
'''
生成Dataset，继承Dataset子类，输入一个txt文件并转为列表格式作为属性
，给出调用index的方法和长度的方法，方法加属性=类可以供DataLoader调用
父类被子类继承，可以重写某些父类的方法和属性，同时继承父类的方法和属性

在 MyDataset 中，主要 获取图片的索引以及定义如何通过索引读取图片及其标签。
'''


class My_Dataset(Dataset):
    def __init__(self,text_path,transform=None,target_transform=None):
        fh = open(text_path,'r')
        imgs = []
        for line in fh:
            line = line.strip()#去掉头尾的值，常规操作
            c = line.split()#根据空格来拆分，对str常规操作
            imgs.append((c[0],int(c[1])))
            #str转换为列表，列表的每个元素有两个值
            
        self.imgs = imgs
        print(len(self.imgs))
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self,index):
        fn,label = self.imgs[index]
        img = Image.open(fn).convert('RGB')#PIL提取图片，因为也适用PIL处理图片
        if self.transform is not None:
            img = self.transform(img)#图片预处理，覆盖形式的。
            
        return img,label
    def __len__(self):
        return len(self.imgs)
    
'''
the way to use:
    train_data = MyDataset(txt_path=train_txt_path, ...)
    train_loader = DataLoader(dataset=train_data, ...) 
    for i, data in enumerate(train_loader, 0) 
    the dataset is called in train_loader,and first instantiate the dataset 
    and the data loader with dataset pull in.
    calling the data in for cirle,and the data in this circle is a list of batch of picture
    '''
    
def Get_cifar10_Data(batch_size):
    
    train_path = '../data/train.txt'
    val_path = '../data/valid.txt'
        

    #define transform
    normMean = (0.4948052, 0.48568845, 0.44682974)
    normStd = (0.24580306, 0.24236229, 0.2603115)
    
    train_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(normMean,normStd)
                                         ])
    val_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(normMean,normStd)
                                        ])
    
    
    #make dataset class by My_dataset and input:dirs of txt,transform
    train_dataset = My_Dataset(train_path,train_transform)
    val_dataset = My_Dataset(val_path,val_transform)
    
    #make DataLoader
    train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)       
    val_loader = DataLoader(val_dataset,batch_size = batch_size,shuffle=True)
    return train_loader,val_loader
if __name__ == '__main__':
    a,b = Get_cifar10_Data(8)
    print(len(a),len(b))
