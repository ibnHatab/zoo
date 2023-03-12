
import os
import time
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision import transforms
# %cd src
from utils import DataModule, HyperParameters, Module, Trainer, add_to_class, show_images

class FMNIST(DataModule):
    def __init__(self, batch_size=256, resize=(28, 28), root='../data'):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
        
    def test_labels(self, indices):
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]
   
    def get_dataloader(self, train):
        dataset = self.train if train else self.val
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                        shuffle=train, num_workers=self.num_workers)

    def visualize(self, batch, num_rows=2, num_cols=8, labels=None):
        X, y = batch
        if not labels:
            labels = self.test_labels(y)
        show_images(X.squeeze(dim=1), num_rows, num_cols, labels)
          
class SGD(HyperParameters):
    """Defined in :numref:`sec_linear_scratch`"""
    def __init__(self, params, lr):
        """Minibatch stochastic gradient descent."""
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            import code; code.interact(local=locals())
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
       
class SoftmaxRegression(Module):    
    def __init__(self, num_inputs, num_outputs, lr, num_gpus=1, sigma=0.01):
        super().__init__(num_gpus=num_gpus)
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)
        if self.gpus:
            self.W = self.W.to(self.gpus[0])        
            self.b = self.b.to(self.gpus[0])
        
    def parameters(self):
        return [self.W, self.b]
    
    def configure_optimizers(self):
        return SGD([self.W, self.b], self.lr)
                
    @staticmethod
    def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdims=True)
        return X_exp / partition
                 
    def forward(self, X):
        #import code; code.interact(local=locals())
        return SoftmaxRegression.softmax(torch.matmul(X.reshape((-1, self.W.shape[0])), self.W) + self.b)
    
    @staticmethod
    def cross_entropy(y_hat, y):
        return - torch.log(y_hat[list(range(len(y_hat))), y]).mean()

    def loss(self, Y_hat, Y):
        return SoftmaxRegression.cross_entropy(Y_hat, Y)
    
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
    
    def accuracy(self, Y_hat, Y, average=True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if average else compare
    
    
data = FMNIST(batch_size=256)    
model = SoftmaxRegression(num_inputs=784, num_outputs=10, lr=0.1, num_gpus=1)
trainer = Trainer(max_epochs=10, num_gpus=1)   

trainer.fit(model, data)


X, y = next(iter(data.train_dataloader()))
data.visualize((X, y))
data.test_labels(y)
preds = model(X)
preds = preds.argmax(axis=1)
data.test_labels(preds)
preds.shape
wrong = preds.type(y.dtype) != y
len(wrong)
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.test_labels(y), data.test_labels(preds))]
data.visualize([X, y], labels=labels)
