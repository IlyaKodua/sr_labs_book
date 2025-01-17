# The script is borrowed from the following repository: https://github.com/clovaai/voxceleb_trainer
# The script creates different loss functions


# Import of modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def accuracy(output, target, topk=(1, )):
    # Computes the precision@k for the specified values of k
    
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0/batch_size))
    
    return res

class SoftmaxLoss(nn.Module):
    # Softmax loss

    def __init__(self, nOut, nClasses, **kwargs):
        super(SoftmaxLoss, self).__init__()

        self.test_normalize = True
        
        self.criterion  = torch.nn.CrossEntropyLoss()
        self.fc         = nn.Linear(nOut, nClasses)

        print('Initialised softmax loss.')

    def forward(self, x, label=None):

        x     = self.fc(x)
        nloss = self.criterion(x, label)
        prec1 = accuracy(x.detach(), label.detach(), topk=(1, ))[0]

        return nloss, prec1

class AMSoftmaxLoss(nn.Module):
    # Additive margin softmax loss

    def __init__(self, nOut, nClasses, margin=0.3, scale=15, **kwargs):
        super(AMSoftmaxLoss, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AM softmax m=%.3f s=%.3f.'%(self.m,self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        
        if label_view.is_cuda: label_view = label_view.cpu()
        
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        
        if x.is_cuda: delt_costh = delt_costh.cuda()
        
        costh_m = costh - delt_costh
        costh_m_s = self.s*costh_m
        loss    = self.ce(costh_m_s, label)
        prec1   = accuracy(costh.detach(), label.detach(), topk=(1, ))[0]
        
        return loss, prec1

class AAMSoftmaxLoss(nn.Module):
    # Additive angular margin softmax loss

    def __init__(self, nOut, nClasses, margin=0.3, scale=15, easy_margin=False, **kwargs):
        super(AAMSoftmaxLoss, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # Make the function cos(theta+m) monotonic decreasing while theta in [0°, 180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m)*self.m

        print('Initialised AAM softmax margin %.3f scale %.3f.'%(self.m,self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine*self.cos_m - sine*self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot*phi) + ((1.0 - one_hot)*cosine)
        output = output*self.s

        loss   = self.ce(output, label)
        prec1  = accuracy(cosine.detach(), label.detach(), topk=(1, ))[0]
        
        return loss, prec1


class AnglProto(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(AnglProto, self).__init__()

        self.test_normalize = True
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2

        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label   = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        nloss   = self.criterion(cos_sim_matrix, label)
        prec1   = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]
        
        return nloss, prec1