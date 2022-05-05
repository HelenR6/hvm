import argparse
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from torchvision import transforms          
import torchvision.models as models
import torch 
import torch.nn as nn
from PIL import Image
import json
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
import torch.nn.functional as F
from load_model import load_model


parser = argparse.ArgumentParser(description='Neural correlation')
# parser.add_argument('--session', type=str)
parser.add_argument('--model_list',nargs="+")
# parser.add_argument('--neuro_wise')

args = parser.parse_args()
# session_name=args.session
# neuro_wise=args.neuro_wise
model_type_list=args.model_list

device='cpu'
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
      
        yield iterable[ndx:min(ndx + n, l)]
        
        
        
#get activation for natural images
import json
from PIL import Image
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torchvision.models as models

for model_type in model_type_list:
  resnet,preprocess=load_model(model_type) 
  images_file = h5py.File('~/projects/def-bashivan/helenr6/hvm_data/hvm_images.h5','r')
  data_file = h5py.File('~/projects/def-bashivan/helenr6/hvm_data/hvm_data.h5','r')
  images_data = images_file['images'][:]
  hvm_data=data_file ['hvm_data'][:]
  v4_index=data_file ['V4_idx'][:]
  v4_data=hvm_data[:,v4_index]

#   session_path=args.session.replace('_','/')
#   final_path=session_path[:-1]+'_'+session_path[-1:]
#   f = h5py.File('/content/gdrive/MyDrive/npc_v4_data.h5','r')
#   natural_data = f['images/naturalistic'][:]
#   synth_data=f['images/synthetic/monkey_'+final_path][:]
#   print(natural_data.shape)

  x = np.array([np.array(preprocess((Image.fromarray(i)).convert('RGB'))) for i in images_data])
  images_tensor=torch.tensor(x)
  print(images_tensor.shape)

#   x = np.array([np.array(preprocess((Image.fromarray(i)).convert('RGB'))) for i in synth_data])
#   synth_perm_tensor=torch.tensor(x)
#   print(synth_perm_tensor.shape)

#   n1 = f.get('neural/naturalistic/monkey_'+final_path)[:]
#   target=np.mean(n1, axis=0)
#   print(target.shape)
#   n2=f.get('neural/synthetic/monkey_'+final_path)[:]
#   neuron_target=np.mean(n2, axis=0)
#   print(neuron_target.shape)

#   neuro_wise=args.neuro_wise
#   print(neuro_wise)
#   if neuro_wise == 'True':
#     with open(f'~/projects/def-bashivan/helenr6/hvm_result/{model_type}_natural_mean.json') as json_file:
#       layerlist=[]
#       load_data = json.load(json_file)
#       json_acceptable_string = load_data.replace("'", "\"")
#       d = json.loads(json_acceptable_string)
#       max_natural_layer=max(d, key=d.get)
#       layerlist.append(max_natural_layer)
  activation={}
  def get_activation(name):
      def hook(model, input, output):
          activation[name] = output.detach()
      return hook
  for layer in layerlist:
    if model_type=="clip":
      exec(f"resnet.visual.{layer}.register_forward_hook(get_activation('{layer}'))")
    elif model_type=="linf_8" or model_type=="linf_4" or model_type=="l2_3" or model_type=='resnet50_l2_eps0.1' or model_type=='resnet50_l2_eps0.01' or model_type=='resnet50_l2_eps0.03' or model_type=='resnet50_l2_eps0.5' or model_type=='resnet50_l2_eps0.25' or model_type=='resnet50_l2_eps3' or model_type=='resnet50_l2_eps5' or model_type=='resnet50_l2_eps1' or model_type=='resnet50_l2_eps0.05':
      exec(f"resnet.model.{layer}.register_forward_hook(get_activation('{layer}'))")
    else:
      exec(f"resnet.{layer}.register_forward_hook(get_activation('{layer}'))")
  counter=0
  for  minibatch in batch(images_tensor,64):
    print(counter)
    if model_type=="clip":
      output=exec(f"resnet.visual(minibatch.to(device))")
    else:
      output=exec(f"resnet(minibatch.to(device))")
    if counter==0:
      with h5py.File(f'{model_type}_natural_layer_activation.hdf5','w')as f:
        for layer in layerlist:
          dset=f.create_dataset(layer,data=activation[layer].cpu().detach().numpy())
    else:
      with h5py.File(f'{model_type}_natural_layer_activation.hdf5','r+')as f:
        for k,v in activation.items():
          print(k)
          data = f[k]
          a=data[...]
          del f[k]
          dset=f.create_dataset(k,data=np.concatenate((a,activation[k].cpu().detach().numpy()),axis=0))
    counter=counter+1

  natural_score_dict={}
  natural_neuron_dict={}
  synth_score_dict={}
  synth_neuron_dict={}
  random_list=[2,10,32,89,43]
  #layerlist=['maxpool','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer4[0]','layer4[1]','layer4[2]','avgpool','fc']
  for key in layerlist:
    natural_score_dict[key]=None
    synth_score_dict[key]=None
  total_synth_corr=[]
  total_natural_corr=[]
  
  with h5py.File(f'{model_type}_natural_layer_activation.hdf5','r')as f:
    # with h5py.File(f'{model_type}_natural_layer_activation.hdf5','r')as f:
      for seed in random_list:
        for k in layerlist:
          print(k)
          natural_data = f[k]
        #   synth_data=s[k]
          a=natural_data[...]
        #   b=synth_data[...]
          pca=PCA(random_state=seed)
          natural_x_pca = pca.fit_transform(torch.tensor(a).cpu().detach().reshape(v4_data.shape[0],-1))
          #synth_x_pca = pca.transform(torch.tensor(b).cpu().detach().reshape(neuron_target.shape[0],-1))
          kfold = KFold(n_splits=5, shuffle=True,random_state=seed)
          # num_neuron=n1.shape[2]
          natural_prediction= np.empty((v4_data.shape[0],v4_data.shape[1]), dtype=object)
          #synth_prediction=np.empty((neuron_target.shape[0],neuron_target.shape[1]), dtype=object)
          for fold, (train_ids, test_ids) in enumerate(kfold.split(natural_x_pca)):
            clf = Ridge(random_state=seed)
            clf.fit((natural_x_pca)[train_ids],v4_data[train_ids])
            start=fold*10
            end=((fold+1)*10)
            natural_prediction[test_ids]=clf.predict((natural_x_pca)[test_ids])
          if natural_score_dict[k] is None:
            natural_corr_array= np.array([pearsonr(natural_prediction[:, i], v4_data[:, i])[0] for i in range(natural_prediction.shape[-1])])
            natural_score_dict[k] = np.median(natural_corr_array)
            total_natural_corr=natural_corr_array
            natural_neuron_dict[k]=total_natural_corr

          else:
            natural_corr_array= np.array([pearsonr(natural_prediction[:, i], v4_data[:, i])[0] for i in range(natural_prediction.shape[-1])])
            natural_score=np.median(natural_corr_array)
            natural_score_dict[k] =np.append(natural_score_dict[k],natural_score)
            total_natural_corr=np.vstack([total_natural_corr,natural_corr_array])
            natural_neuron_dict[k]=total_natural_corr

          print(natural_score_dict[k])


    #   if neuro_wise=='True':
      np.save(f'~/projects/def-bashivan/helenr6/hvm_result/{model_type}_synth_neuron_corr.npy',total_synth_corr)
      np.save(f'~/projects/def-bashivan/helenr6/hvm_result/{model_type}_natural_neuron_corr.npy',total_natural_corr)

      from statistics import mean
      new_natural_score_dict = {k:  v.tolist() for k, v in natural_score_dict.items()}
      new_synth_score_dict = {k:  v.tolist() for k, v in synth_score_dict.items()}
      import json
      
      natural_json = json.dumps(new_natural_score_dict, indent = 4) 
      print(natural_json)
        
      with open(f"~/projects/def-bashivan/helenr6/hvm_result/{model_type}_natural.json", 'w') as f:
        json.dump(natural_json, f)
        
      natural_mean_dict = {k:  mean(v) for k, v in natural_score_dict.items()}
        
      json_object = json.dumps(natural_mean_dict, indent = 4) 
      print(json_object)
      with open(f"~/projects/def-bashivan/helenr6/hvm_result/{model_type}_natural_mean.json", 'w') as f:
        json.dump(json_object, f)








