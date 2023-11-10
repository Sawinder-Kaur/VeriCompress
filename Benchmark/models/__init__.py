from models.resnet import model_resnet
from models.feedforward import *
from models.resnext import *
from models.resnext_imagenet64 import *
from models.densenet import *
from models.mobilenet import *
from models.densenet_no_bn import *
from models.densenet_imagenet import *
from models.wide_resnet_imagenet64 import *
from models.wide_resnet_cifar import *
from models.resnet18 import *
from models.vnncomp_resnet import resnet2b as vnncomp_resnet2b, resnet4b as vnncomp_resnet4b
from models.resnet_small import *
from models.vgg_mnist import *
from models.vgg_cifar import *


Models = {
    'mlp_2layer': mlp_2layer,
    'mlp_3layer': mlp_3layer,
    'mlp_3layer_weight_perturb': mlp_3layer_weight_perturb,
    'mlp_5layer': mlp_5layer,
    'cnn_4layer': cnn_4layer,
    'cnn_6layer': cnn_6layer,
    'cnn_7layer': cnn_7layer,
    'cnn_7layer_bn': cnn_7layer_bn,
    'cnn_7layer_bn_imagenet': cnn_7layer_bn_imagenet,
    'cnn_16layer': cnn_16layer,
    'resnet': model_resnet,
    'resnet5':resnet5,
    'resnet9':resnet9,
    'resnet5_mnist':resnet5_mnist,
    'resnet9_mnist':resnet9_mnist,
    'resnet18': ResNet18,
    'resnet18_mnist': ResNet18_mnist,
    'ResNeXt_cifar': ResNeXt_cifar,
    'ResNeXt_imagenet64': ResNeXt_imagenet64,
    'Densenet_cifar_32': Densenet_cifar_32,
    'Densenet_cifar_wobn': Densenet_cifar_wobn,
    'Densenet_imagenet': Densenet_imagenet,
    'MobileNet_cifar': MobileNetV2,
    'wide_resnet_cifar': wide_resnet_cifar,
    'wide_resnet_cifar_bn': wide_resnet_cifar_bn,
    'wide_resnet_cifar_bn_wo_pooling': wide_resnet_cifar_bn_wo_pooling,
    'wide_resnet_cifar_bn_wo_pooling_dropout': wide_resnet_cifar_bn_wo_pooling_dropout,
    'wide_resnet_imagenet64': wide_resnet_imagenet64,
    'wide_resnet_imagenet64_1000class': wide_resnet_imagenet64_1000class,
    'vnncomp_resnet2b': vnncomp_resnet2b,
    'vnncomp_resnet4b': vnncomp_resnet4b,
    'vgg16_m':vgg16_m,
    'vgg16_bn_m':vgg16_bn_m,
    'vgg16':vgg16,
}
