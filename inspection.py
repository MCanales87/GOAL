import pprint
import torch

print('MNET : ');

mnetmodel = torch.load('models/MNet_model.pt', map_location="cuda");

cantmodel=len(list(mnetmodel.items())); 

for i in range(cantmodel):
    print(str(i)+ '- ' + list(mnetmodel.items())[i][0]+'\r\t\t\t\t\t', end='\t');
    print(list(mnetmodel.items())[i][1].shape, end='\t');
    print('\r\t\t\t\t\t\t\t\t\t[ ', end='');
    print(round(torch.min(list(mnetmodel.items())[i][1]).item(),2), end='');
    print(', ', end='');
    print(round(torch.max(list(mnetmodel.items())[i][1]).item(),2), end=' ]\n');


print('\n\nGNET : ');

getmodel = torch.load('models/GNet_model.pt', map_location="cuda");

cantmodel=len(list(getmodel.items())); 

for i in range(cantmodel):
    print(str(i)+ '- ' + list(getmodel.items())[i][0]+'\r\t\t\t\t\t', end='\t');
    print(list(getmodel.items())[i][1].shape, end='\t');
    print('\r\t\t\t\t\t\t\t\t\t[ ', end='');
    print(round(torch.min(list(getmodel.items())[i][1]).item(),2), end='');
    print(', ', end='');
    print(round(torch.max(list(getmodel.items())[i][1]).item(),2), end=' ]\n');






