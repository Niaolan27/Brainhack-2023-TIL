import numpy as np
import torch
import torch.nn.functional as F

def infer(model, img, target, transform = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if transform is not None:
      output1, output2 = transform(img).unsqueeze(0), transform(target).unsqueeze(0)
    else:
      output1, output2 = img, target
    #generate the embedding vectors using loaded model
    output1,output2,_ = model(output1.to(device),output2.to(device)) 

    #calculating euclidean distance and determining if match
    euclidean_distance = F.pairwise_distance(output1, output2)
    print(f'eucl {euclidean_distance}')

    pred = torch.Tensor(np.array([1 if dist < 1 else 0 for dist in euclidean_distance]))

    return pred, euclidean_distance
