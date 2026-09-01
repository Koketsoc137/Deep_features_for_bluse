import torch
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import kornia.augmentation as K
import kornia
import random
import time
import numpy as np
import skdim
#import VISUAL as viz


meerkat_dir = "/idia/projects/hippo/Koketso/meerkat"
dogbreed_dir = "/idia/projects/hippo/Koketso/dog_breeds"
galaxyzoo_dir = "/idia/projects/hippo/Koketso/galaxy_zoo_sub"
hand_dir = "/idia/projects/hippo/Koketso/Train_Alphabet"

class Custom(Dataset):
    def __init__(self,x,names,resize = 300,crop = 224,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225], transform = None):
        self.x = x
        self.resize = resize
        self.crop = crop
        self.mean = mean
        self.std = std
        self.name = names
        
        self.transform = tv.transforms.Compose([
                            #tv.transforms.ToPILImage(),
                            #tv.transforms.Resize((424,424)),
                            tv.transforms.Resize(self.resize),
                            tv.transforms.CenterCrop(self.crop),          
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean=self.mean, std=self.std)
                            ])
        if transform != None:
            self.transform = transform
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        
        image = self.x[index]
        name = self.name[index]
        
        #image = np.array(image[0].getdata())
        #plt.imshow(image)
        
        
        #image = self.tv.transforms.ToTensor(image)
        image = tv.transforms.CenterCrop(self.crop)(image[0])
        x = self.transform(image)
        # defined the transform below
        return x, name
            
class RandomRotationWithCrop(K.RandomRotation):
    def __init__(self, degrees, crop_size, output_dim = 244,p = 0.5):
        super(RandomRotationWithCrop, self).__init__(degrees, p = 1)
        #super(RandomRotationWithCrop, self).__init__(crop_size)
        self.rotation_transform = K.RandomRotation(degrees)
        self.crop_transform = K.CenterCrop(crop_size, keepdim = False,align_corners = True)
        self.resize_transform = K.Resize(output_dim)
    def __call__(self, x):
        if random.random() <self.p:
            # Apply random rotation
            x = self.rotation_transform(x)

            # Apply center crop
            x = self.crop_transform(x)
            x = self.resize_transform(x)

        return x
    
class Custom_labelled(Dataset):
    def __init__(self,dataset,names,resize = 300,crop = 224,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        self.data = dataset
        self.name = names
        self.resize = resize
        self.crop = crop
        self.mean = mean
        self.std = std
        self.transform = tv.transforms.Compose([
                            #tv.transforms.ToPILImage(),
                            #tv.transforms.Resize((424,424)),
                            tv.transforms.Resize(self.resize),
                            tv.transforms.CenterCrop(self.crop),          
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean=self.mean, std=self.std),


                            ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        
        datapoint = self.data[index]
        name = self.name[index]
        
        #image = np.array(image[0].getdata())
        #plt.imshow(image)
        
        #image = self.tv.transforms.ToTensor(image)
        image = self.transform(datapoint[0])
        label = datapoint[1]
        
        # defined the transform below
        return image,label,name
    
    
class Custom_labelled_pandas(torch.utils.data.Dataset):
    def __init__(self,dataframe,resize = 300,crop = 224,mean=[0.5, 0.5, 0.5],std=[0.2, 0.2, 0.2]):

        self.resize = resize
        self.crop = crop
        self.mean = mean
        self.std = std
        self.dataframe = dataframe
        self.transform = tv.transforms.Compose([
                            #tv.transforms.ToPILImage(),
                            #tv.transforms.Resize((424,424)),
                            tv.transforms.Resize(self.resize),
                            tv.transforms.CenterCrop(self.crop),          
                            tv.transforms.ToTensor(),
                            tv.transforms.Grayscale(num_output_channels = 3),
                            tv.transforms.Normalize(mean=self.mean, std=self.std)
                            ])
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self,index):
        
        image = self.x[index]
        target = self.y[index]
        
        #image = np.array(image[0].getdata())
        #plt.imshow(image)
        
        #image = self.tv.transforms.ToTensor(image)
        x = self.transform(image[0])
        
        # defined the transform below
        return x,target
class ArrayDataset(Dataset):
    def __init__(
        self,
        images,
        labels=None,
        names=None,
        resize=256,
        crop=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        eval_mode=False,
    ):
        self.images = images
        self.labels = labels
        self.names = names

        self.train_transform = tv.transforms.Compose([
            tv.transforms.Resize(resize),
            tv.transforms.CenterCrop(crop),
            tv.transforms.RandomResizedCrop(size=crop, scale=(0.7, 1.0)),
            tv.transforms.RandomRotation((0, 360)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=std),
        ])

        self.eval_transform = tv.transforms.Compose([
            tv.transforms.Resize(resize),
            tv.transforms.CenterCrop(crop),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=std),
        ])

        self.eval_mode = eval_mode

    @property
    def transform(self):
        return self.eval_transform if self.eval_mode else self.train_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx], self.names[idx]
        return image
    
    
def dataset(data):
    if data == 'meerkat':
        Dir = "/idia/projects/hippo/Koketso/meerkat"
    elif data =="dog_breed":
        Dir = "/idia/projects/hippo/Koketso/dog_breeds"
    elif data == "galaxy_zoo":
        Dir = "/idia/projects/hippo/Koketso/galaxy_zoo_sub"
    elif data == "hand_alphabet":
        Dir = "/idia/projects/hippo/Koketso/Train_Alphabet"
    else:
        Dir = data
    return tv.datasets.ImageFolder(Dir)

def transformed(dataset):
    return Custom(dataset) 


def plot_filters_single_channel(t):

    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12

    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    plt.show()

def plot_filters_multi_channel(t):

    #get the number of kernals
    num_kernels = t.shape[0]

    #define number of columns for subplots
    num_cols = 12
    #rows = num of kernels
    num_rows = num_kernels

    #set the figure size
    fig = plt.figure(figsize=(num_cols,num_rows))

    #looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)

        #for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        #standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.savefig('myimage.png', dpi=100)
    plt.tight_layout()
    plt.show()


def plot_weights(model, layer_num, single_channel = True, collated = False):

  #extracting the model features at the particular layer number
  layer = model.features[layer_num]

  #checking whether the layer is convolution layer or not
  if isinstance(layer, torch.nn.Conv2d):
    #getting the weight tensor data
    weight_tensor = model.features[layer_num].weight.data

    if single_channel:
      if collated:
        plot_filters_single_channel_big(weight_tensor)
      else:
        plot_filters_single_channel(weight_tensor)

    else:
      if weight_tensor.shape[1] == 3:
        plot_filters_multi_channel(weight_tensor)
      else:
        print("Can only plot weights with three channels with single channel = False")

  else:
    print("Can only visualize layers which are convolutional")
    
def train_val_dataset(dataset, source_ids = None,val_split=0.30,train_size = None):
    
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split,train_size = train_size, random_state = 42)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    if source_ids is not None:
        datasets['train_ids'] = Subset(source_ids, train_idx)
        datasets['val_ids'] = Subset(source_ids, val_idx)

    return datasets

def features(loader,model,named = True,batch_size = 128,device = torch.device('cuda:0'), patch_level_features = True):
    
    
    time1 = time.time()
    rep = []
    labells = []
    names = []
    images = []
    name = "_"
    label = 0
    i = 0
    model.eval()
    print("validating")
    with torch.no_grad():
        if named:
            
            for image,label,name in loader:                                   #name
                if i*batch_size > 100000:
                    break;
                #images.append(image)
                image = image.to(device)
                feature = model(image)
                rep.append(feature)
                labells.append(label)
                if i%10==0:
                    print(i*batch_size)
                names.append(name)                      #name
                i+=1
        else:
                
            for image,label in loader:                                   #name
                if i*batch_size > 100000:
                    break;
                #images.append(image)
                image = image.to(device)
                rep.append(model(image))
                
                labells.append(label)

                i+=1

    #Unwrappping the data
    rep2 = []
    labells2 = []
    rep2 = []
    images2 = []



    for i in range(len(rep)):
        for j in range(len(rep[i])):
            #images2.append(images[i][j].cpu().numpy()) #Images
            rep_ = rep[i][j].cpu().numpy()
            label_ = labells[i][j].item()

            if patch_level_features:
                len_rep = len(rep_)
                len_patch_rep = len(rep_[0])
                rep_ =  np.reshape(rep_,(len_rep*len_patch_rep))
            rep2.append(rep_)        #Representations
            labells2.append(labells[i][j].item())


    rep = rep2
    if patch_level_features:
        rep = viz.pca(data = rep,variance = 0.95,n_components =700)
        #images = images2 
    labels = labells2
    model.train()

    return rep,labels


def get_representations(model = None,
                        loader = None, 
                        batch_size = 128,
                        patch_level_features = False,
                        epoch = 0,
                        labeled = True,
                        device = "cuda",
                       encoder = False):
    #initialise global variable
    rep = []
    """
    def hook_fn(module, input, output):
        flat_output = output.view(output.size(0), -1).cpu()
        print(TwoNN.twonn(flat_output,plot = False))
        rep.append(flat_output)
    #model.to(device) model already  on device
    """
    #hook = model.avgpool.register_forward_hook(hook_fn)
    if not torch.cuda.is_available():
        device = "cpu" 
    if not encoder:
        model.classifier[-1] = torch.nn.Identity()
    model.eval()

    # send model to cude
    
    if not next(model.parameters()).is_cuda:
        model.to(device)
    


    # representations
    rep = []
    labels = []
    names = []
    label = 0
    with torch.no_grad():
            
        for batch in loader:                                   
            if len(batch) ==3:
                image,label,name = batch
                labeled =  True
            else:
                image, name = batch
            
            image = image.to(device)
            output = model(image).cpu()
            #Id = TwoNN.twonn(output,plot = False)[0][0].item()
            labels.append(label)
            rep.append(output)
            names.append(name)
            torch.cuda.empty_cache()
    #twoNNs = np.array(twoNNs)
    #hook.remove()
    rep2 = []
    labels2 = []
    names2 = []

    for i in range(len(rep)):
        for j in range(len(rep[i])):
            #images2.append(images[i][j].cpu().numpy()) #Images
            rep_ = rep[i][j].numpy()
            name_  = names[i][j]

            if labeled:
                label_ = labels[i][j].item()
                labels2.append(label_)
            else:

                labels2.append(0)
            names2.append(name_)
    

            rep2.append(rep_)        #Representations

    return rep2, labels2, names2

    #umap = viz.umap(rep2,name = "Features on epoch:"+str(epoch))
    #pca = viz.pca(rep2,variance = 0.95, return_n_components = 20)
    #pca = pca/np.max(pca)
    #print("Correlating and plotting")
    #AstroMLmod.correlate_and_plot(umap,min_dist = 0.0, max_dist =2.5, label = "Correlation on umap epoch:"+str(epoch))
    #viz.pca(rep2,variance = 0.95)
    #print(np.mean(twoNNs, axis = 0),np.std(twoNNs,axis = 0, ddof=1))
    





#visualize weights for Alexnet - first conv layer
#plot_weights(resnet50, 0, single_channel = False)

    
