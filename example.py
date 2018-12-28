from benchmark import benchmarking
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
import os
import sys
data_dir = os.environ['TESTDATADIR']
assert data_dir is not None, "No data directory"

model = resnet18(pretrained=True)
@benchmarking(team=12, task=0, model=model, preprocess_fn=None)
def inference(net, data_loader,**kwargs):
    total = 0
    correct = 0
    assert kwargs['device'] != None, 'Device error'
    device = kwargs['device']
    model.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc

if __name__=='__main__':
    transform_test = transforms.Compose([
    transforms.Resize((224,224),),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    inference(model, testloader)
