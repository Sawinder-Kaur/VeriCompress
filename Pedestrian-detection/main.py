import argparse
import multiprocessing
import random
import time
import logging
import os
import glob
import pandas as pd 
import xml.etree.ElementTree as ET
import cv2
from sklearn import preprocessing

import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss

from feedforward import cnn_7layer,cnn_16layer,cnn_4layer
from auto_LiRPA import BoundedModule, BoundedTensor, BoundDataParallel, CrossEntropyWrapper
from auto_LiRPA.bound_ops import BoundExp
from auto_LiRPA.eps_scheduler import LinearScheduler, SmoothedScheduler, AdaptiveScheduler, FixedScheduler
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter, logger, get_spec_matrix



from utils import deactivate,  remove_redundant_nodes

def get_exp_module(bounded_module):
    for _, node in bounded_module.named_modules():
        # Find the Exp neuron in computational graph
        if isinstance(node, BoundExp):
            return node
    return None

parser = argparse.ArgumentParser()

parser.add_argument("--verify", action="store_true", help='verification mode, do not train')
parser.add_argument("--no_loss_fusion", action="store_true", help='without loss fusion, slower training mode')
parser.add_argument("--load", type=str, default="", help='Load pretrained model')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument("--data", type=str, default="PD", choices=["MNIST", "CIFAR","PD"], help='dataset')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--eps", type=float, default=2/255, help='Target training epsilon')
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN"], help='method of bound analysis')
parser.add_argument("--model", type=str, default="cnn_7layer_bn",
                    help='model name (Densenet_cifar_32, resnet18, ResNeXt_cifar, MobileNet_cifar, wide_resnet_cifar_bn_wo_pooling)')
parser.add_argument("--num_epochs", type=int, default=330, help='number of total epochs')
parser.add_argument("--batch_size", type=int, default=20, help='batch size')
parser.add_argument("--lr", type=float, default=5e-4, help='learning rate')
parser.add_argument("--lr_decay_rate", type=float, default=0.1, help='learning rate decay rate')
parser.add_argument("--lr_decay_milestones", nargs='+', type=int, default=[250, 300], help='learning rate dacay milestones')
parser.add_argument("--scheduler_name", type=str, default="SmoothedScheduler",
                    choices=["LinearScheduler", "SmoothedScheduler"], help='epsilon scheduler')
parser.add_argument("--scheduler_opts", type=str, default="start=15,length=150,mid=0.4", help='options for epsilon scheduler')
parser.add_argument("--bound_opts", type=str, default=None, choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options for relu')
parser.add_argument('--clip_grad_norm', type=float, default=8.0)
parser.add_argument("--pre", type = str, default = "")
parser.add_argument("--parameter_budget", type=float, default=100000)
parser.add_argument("--thickening_length", type=int, default=1)
parser.add_argument("--compress", type = str, default = "True")
parser.add_argument("--verbose", type = str, default = "False")
parser.add_argument("--deploy",type=str,default = "False")

args = parser.parse_args()
exp_name = args.model + '_b' + str(args.batch_size) + '_' + str(args.bound_type) + '_epoch' + str(args.num_epochs) + '_' + args.scheduler_opts + '_' + str(args.eps)[:6]
os.makedirs('saved_models/', exist_ok=True)
log_file = f'saved_models/{exp_name}{"_test" if args.verify else ""}.log'
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)


def compute_gradient_norm(model):
    gradients = []
    for param in model.parameters():
        gradients.append(np.sum(np.abs(param.grad.data.detach().cpu().numpy())))
    gradient_sum = sum(gradients)
    return gradient_sum


def Train(model,model_ori, t, loader, eps_scheduler, norm, train, opt, bound_type, method='robust', loss_fusion=True, final_node_name=None):
    num_class = 2
    meter = MultiAverageMeter()
    if train:
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
    else:
        model.eval()
        eps_scheduler.eval()

    exp_module = get_exp_module(model)
    gradient_sum = 0
    batches = 0
    def get_bound_loss(x=None, c=None):
        if loss_fusion:
            bound_lower, bound_upper = False, True
        else:
            bound_lower, bound_upper = True, False

        if bound_type == 'IBP':
            lb, ub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None, final_node_name=final_node_name, no_replicas=True)
        elif bound_type == 'CROWN':
            lb, ub = model(method_opt="compute_bounds", x=x, IBP=False, C=c, method='backward',
                                          bound_lower=bound_lower, bound_upper=bound_upper)
        elif bound_type == 'CROWN-IBP':
            # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method='backward')  # pure IBP bound
            # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
            factor = (eps_scheduler.get_max_eps() - eps_scheduler.get_eps()) / eps_scheduler.get_max_eps()
            ilb, iub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None, final_node_name=final_node_name, no_replicas=True)
            if factor < 1e-50:
                lb, ub = ilb, iub
            else:
                clb, cub = model(method_opt="compute_bounds", IBP=False, C=c, method='backward',
                             bound_lower=bound_lower, bound_upper=bound_upper, final_node_name=final_node_name, no_replicas=True)
                if loss_fusion:
                    ub = cub * factor + iub * (1 - factor)
                else:
                    lb = clb * factor + ilb * (1 - factor)

        if loss_fusion:
            if isinstance(model, BoundDataParallel):
                max_input = model(get_property=True, node_class=BoundExp, att_name='max_input')
            else:
                max_input = exp_module.max_input
            return None, torch.mean(torch.log(ub) + max_input)
        else:
            # Pad zero at the beginning for each example, and use fake label '0' for all examples
            lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
            fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
            robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
            return lb, robust_ce

    for i, (data, labels) in enumerate(loader):
        batches +=1
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = method
        if eps < 1e-50:
            batch_method = "natural"
        if train:
            opt.zero_grad()
        # bound input for Linf norm used only
        if norm == np.inf:
            data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / loader.std).view(1,-1,1,1), data_max)
            data_lb = torch.max(data - (eps / loader.std).view(1,-1,1,1), data_min)
        else:
            data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data, labels = data.cuda(), labels.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
        x = BoundedTensor(data, ptb)
        if loss_fusion:
            if batch_method == 'natural' or not train:
                output = model(x, labels)  # , disable_multi_gpu=True
                regular_ce = torch.mean(torch.log(output))
            else:
                model(x, labels)
                regular_ce = torch.tensor(0., device=data.device)
            meter.update('CE', regular_ce.item(), x.size(0))
            x = (x, labels)
            c = None
        else:
            # Generate speicification matrix (when loss fusion is not used).
            c = get_spec_matrix(data, labels, num_class)
            x = (x,) if final_node_name is None else (x, labels)
            output = model(x, final_node_name=final_node_name)
            regular_ce = CrossEntropyLoss()(output, labels)  # regular CrossEntropyLoss used for warming up
            meter.update('CE', regular_ce.item(), x[0].size(0))
            meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).item() / x[0].size(0), x[0].size(0))

        if batch_method == 'robust':
            lb, robust_ce = get_bound_loss(x=x, c=c)
            loss = robust_ce
        elif batch_method == 'natural':
            loss = regular_ce
        if train:
            loss.backward()
            gradient_sum += compute_gradient_norm(model_ori)
            if args.clip_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                meter.update('grad_norm', grad_norm)

            if isinstance(eps_scheduler, AdaptiveScheduler):
                eps_scheduler.update_loss(loss.item() - regular_ce.item())
            opt.step()
        meter.update('Loss', loss.item(), data.size(0))

        if batch_method != 'natural':
            meter.update('Robust_CE', robust_ce.item(), data.size(0))
            if not loss_fusion:
                # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
                # If any margin is < 0 this example is counted as an error
                meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
        meter.update('Time', time.time() - start)

        if (i + 1) % 50 == 0 and train:
            logger.info('[{:2d}:{:4d}]: eps={:.12f} {}'.format(t, i + 1, eps, meter))
    
    
    logger.info('[{:2d}:{:4d}]: eps={:.12f} {}'.format(t, i + 1, eps, meter))
    if train == True and args.verbose == "True": 
        #count_parameters(model)
        print("Average gradient sum : {}".format(gradient_sum/batches))
    return meter

def creatingInfoData(Annotpath):
    information={'xmin':[],'ymin':[],'xmax':[],'ymax':[],'ymax':[],'name':[]
                ,'label':[]}

    for file in sorted(glob.glob(str(Annotpath+'/*.xml*'))):
        dat=ET.parse(file)
        for element in dat.iter():    

            if 'object'==element.tag:
                for attribute in list(element):
                    if 'name' in attribute.tag:
                        name = attribute.text                 
                        information['label'] += [name]
                        information['name'] +=[file.split('/')[-1][0:-4]]

                    if 'bndbox'==attribute.tag:
                        for dim in list(attribute):
                            if 'xmin'==dim.tag:
                                xmin=int(round(float(dim.text)))
                                information['xmin']+=[xmin]
                            if 'ymin'==dim.tag:
                                ymin=int(round(float(dim.text)))
                                information['ymin']+=[ymin]
                            if 'xmax'==dim.tag:
                                xmax=int(round(float(dim.text)))
                                information['xmax']+=[xmax]
                            if 'ymax'==dim.tag:
                                ymax=int(round(float(dim.text)))
                                information['ymax']+=[ymax]
                     
    return pd.DataFrame(information)

def croppingFromImage(path,Data_information):
    cropped_image=[]
    label=[]
    for i in range(0,len(Data_information)):
        img=cv2.imread(path+'/'+Data_information['name'][i]+'.jpg',cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_=img[Data_information['ymin'][i]:Data_information['ymax'][i],Data_information['xmin'][i]:Data_information['xmax'][i]]
        cropped_image.append(img_)
        
        label.append(Data_information['label'][i])
    return cropped_image , label   

def resizing(data,size):
        resizing=[]
        for i in data:
            resizing.append(cv2.resize(i,(size,size)))

        return resizing 


    



def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_path=r'Train/Train/JPEGImages'
    train_annot=r'Train/Train/Annotations'

    test_path=r'Test/Test/JPEGImages'
    test_annot=r'Test/Test/Annotations'

    val_path=r'Val/Val/JPEGImages'
    val_annot=r'Val/Val/Annotations'

    train_info=creatingInfoData(train_annot)
    test_info=creatingInfoData(test_annot)
    val_info=creatingInfoData(val_annot)

    trainImage , trainLabel =croppingFromImage(train_path,train_info)
    testImage , testLabel =croppingFromImage(test_path,test_info)

    valImage,valLabel=croppingFromImage(val_path,val_info)
    print(len(trainImage) == len(trainLabel))
    print(len(testImage) == len(testLabel))
    print(len(valImage) == len(valLabel))
    trainShapes=[]
    testShapes=[]
    for i in trainImage :
        trainShapes.append(i.shape)
    for i in testImage :
        testShapes.append(i.shape)

    pd.Series(trainShapes).value_counts()[:10]
    pd.Series(testShapes).value_counts()[:10]
     
    s=140
    X_train,X_test,X_Val=resizing(trainImage,s),resizing(testImage,s),resizing(valImage,s)
    labeling=preprocessing.LabelEncoder()
    y_train=labeling.fit_transform(trainLabel)
    y_test=labeling.fit_transform(testLabel)
    y_val=labeling.fit_transform(valLabel)
    labeling.classes_
    
    X_train,X_test,X_Val=np.asarray(X_train).astype(np.float32),np.asarray(X_test).astype(np.float32), np.asarray(X_Val).astype(np.float32)
    X_train,X_test,X_Val=X_train/255.0,X_test/255.0,X_Val/255.0
    #print(numpy.size(X_train))
    #print(trainLabel)
    #X_train,X_test=np.reshape(X_train,(-1,3,s,s)),np.reshape(X_test,(-1,3,s,s))
    
    X_train,X_test = torch.from_numpy(X_train),torch.from_numpy(X_test)
    y_train,y_test = torch.from_numpy(y_train),torch.from_numpy(y_test)
    X_train = X_train.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)
    
    print(X_train.size())
    train_data = torch.utils.data.TensorDataset(X_train,y_train)
    test_data = torch.utils.data.TensorDataset(X_test,y_test)
    print(train_data)
    
    
    model_ori = cnn_7layer(in_ch=3, in_dim=140)
    #model_mask = cnn_7layer(in_ch=3, in_dim=140)
    print(model_ori)
    ## Step 2: Prepare dataset as usual
    
    dummy_input = torch.randn(2, 3, s, s)
    #print("Original_model : {}".format(args.original_model))
    #pruned_model_name, finetuned_model_name = model_names(args)
    #print("Pruned_model_name : {}".format(pruned_model_name))
    #print("Sparse_model_name : {}".format(sparse_model_name))
    #print("Finetuned_model_name : {}".format(finetuned_model_name))


    train_data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    test_data = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    
    """
    for sample,label in test_data:
        print(sample.size())
    """
    train_data.mean = test_data.mean = torch.tensor([0.0])
    train_data.std = test_data.std = torch.tensor([1.0])

    """ 
    if args.prune == True and args.random_init == "False":
        model_ori.load_state_dict(torch.load(args.original_model,map_location=args.device)['state_dict'])
    """
    #acc_orig = test_accuracy(model_ori, test_data,args.device)
    #print("Test accuracy of the original model : {}".format(acc_orig))
    
    model = BoundedModule(model_ori, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)
    final_name1 = model.final_name
    model_loss = BoundedModule(CrossEntropyWrapper(model_ori), (dummy_input, torch.zeros(1, dtype=torch.long)),
                               bound_opts={'relu': args.bound_opts, 'loss_fusion': True}, device=args.device)
    # after CrossEntropyWrapper, the final name will change because of one additional input node in CrossEntropyWrapper
    final_name2 = model_loss._modules[final_name1].output_name[0]
    assert type(model._modules[final_name1]) == type(model_loss._modules[final_name2])
    if args.no_loss_fusion:
        model_loss = BoundedModule(model_ori, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)
        final_name2 = None
    model_loss = BoundDataParallel(model_loss)

    
    opt = optim.Adam(model_loss.parameters(), lr=args.lr)
    norm = float(args.norm)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_decay_milestones, gamma=args.lr_decay_rate)
    eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)
    logger.info(str(model_ori))
    
    if args.compress == "True": 
        deactivate(model_ori, args)
    ## Step 5: start training
    if args.verify:
        eps_scheduler = FixedScheduler(args.eps)
        with torch.no_grad():
            Train(model,model, 1, test_data, eps_scheduler, norm, False, None, 'IBP', loss_fusion=False, final_node_name=None)
    else:
        timer = 0.0
        best_err = 1e10
        # with torch.autograd.detect_anomaly():
        for t in range(1,args.num_epochs+1):
            logger.info("Epoch {}, learning rate {}".format(t, lr_scheduler.get_last_lr()))
            start_time = time.time()
            Train(model_loss,model, t, train_data, eps_scheduler, norm, True, opt, args.bound_type, loss_fusion=not args.no_loss_fusion)
            lr_scheduler.step()
            epoch_time = time.time() - start_time
            timer += epoch_time
            logger.info('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))

            logger.info("Evaluating...")
            torch.cuda.empty_cache()

            # remove 'model.' in state_dict for CrossEntropyWrapper
            state_dict_loss = model_loss.state_dict()
            state_dict = {}
            if not args.no_loss_fusion:
                for name in state_dict_loss:
                    assert (name.startswith('model.'))
                    state_dict[name[6:]] = state_dict_loss[name]
            else:
                state_dict = state_dict_loss

            with torch.no_grad():
                if t > int(eps_scheduler.params['start']) + int(eps_scheduler.params['length']):
                    m = Train(model_loss,model, t, test_data, FixedScheduler(args.eps), norm, False, None, 'IBP', loss_fusion=False,
                              final_node_name=final_name2)
                else:
                    m = Train(model_loss,model, t, test_data, eps_scheduler, norm, False, None, 'IBP', loss_fusion=False, final_node_name=final_name2)
            if args.compress == "True" and t% args.thickening_length == 0: 
                deactivate(model_ori, args)
                with torch.no_grad():
                    print("After Pruning")
                    if t > int(eps_scheduler.params['start']) + int(eps_scheduler.params['length']):
                        m = Train(model_loss,model, t, test_data, FixedScheduler(args.eps), norm, False, None, 'IBP', loss_fusion=False,
                              final_node_name=final_name2)
                    else:
                        m = Train(model_loss,model, t, test_data, eps_scheduler, norm, False, None, 'IBP', loss_fusion=False, final_node_name=final_name2)
            save_dict = {'state_dict': state_dict, 'epoch': t, 'optimizer': opt.state_dict()}
            if t < int(eps_scheduler.params['start']):
                torch.save(save_dict, 'saved_models/natural_' + exp_name)
            elif t > int(eps_scheduler.params['start']) + int(eps_scheduler.params['length']):
                current_err = m.avg('Verified_Err')
                if current_err < best_err:
                    best_err = current_err
                    torch.save(save_dict, 'saved_models/' + exp_name + '_best_' + str(best_err)[:6])
                else:
                    torch.save(save_dict, 'saved_models/' + exp_name)
            else:
                torch.save(save_dict, 'saved_models/' + exp_name)
            torch.cuda.empty_cache()
            finetuned_model_name = args.pre  + args.model + "_" + args.data + "_" + str(args.eps) +"_"+ str(int(args.parameter_budget))+ ".pt"    
            torch.save({'state_dict': model_ori.state_dict(), 'epoch': t}, finetuned_model_name if finetuned_model_name != "" else args.model)
        if args.prune_mechanism == "structured" and args.deploy == "True": 
            print("Removing Redundant Parameters ...")
            dummy_input = dummy_input.to(args.device)
            remove_redundant_nodes(model_ori,dummy_input,args.parameter_budget)
            reduced_model_name = "reduced_" + args.pre  + args.model + "_" + args.data + "_" + str(args.eps) +"_"+ str(int(args.parameter_budget))+ ".pt" 
            torch.save({'state_dict': model_ori.state_dict(), 'epoch': t}, reduced_model_name if finetuned_model_name != "" else args.model)
            print("Model saved as {}".format(reduced_model_name))

if __name__ == "__main__":
    logger.info(args)
    main(args)