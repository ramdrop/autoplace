# %% pytorch
import torch
import argparse
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# public library
import logging
from datetime import datetime
import os
import sys
import numpy as np
import tqdm
import h5py
import json
import shutil
# from torchsummary import summary
import importlib
import random
import atexit
os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# private library
import nuscene as dataset
import evaluate as evaluate

class FixRandom():
    def __init__(self, seed) -> None:
        self.seed = seed
        self.set_everything_fixed()

    def set_everything_fixed(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    def seed_worker(self, worker_id):
        # worker_seed = torch.initial_seed() % 2**32
        worker_seed = self.seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def quick_log(logfile, *args):
    with open(os.path.join(opt.runsPath, logfile), 'a') as f:
        for arg in args:
            f.write(arg)
            f.flush()
            print(arg, end='')


def update_opt_from_json(flag_file, opt):
    restore_var = ['net', 'seqLen', 'num_clusters', 'output_dim', 'structDir', 'imgDir', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 'num_clusters', 'optim', 'margin', 'seed', 'patience']
    # flag_file = os.path.join(opt.resume, 'flags.json')
    if os.path.exists(flag_file):
        with open(flag_file, 'r') as f:
            stored_flags = {'--' + k: str(v) for k, v in json.load(f).items() if k in restore_var}
            to_del = []
            for flag, val in stored_flags.items():
                for act in parser._actions:
                    if act.dest == flag[2:]:
                        # store_true / store_false args don't accept arguments, filter these
                        if type(act.const) == type(True):
                            if val == str(act.default):
                                to_del.append(flag)
                            else:
                                stored_flags[flag] = ''
            for flag in to_del:
                del stored_flags[flag]

            train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
            print('restored flags:', train_flags)
            opt = parser.parse_args(train_flags, namespace=opt)
    return opt


def evaluate_model(opt, seed_worker=None,):
    # load configurations
    opt.runsPath = opt.resume
    print('resume path:', opt.resume)
    opt = update_opt_from_json(os.path.join(opt.resume, 'flags.json'), opt)

    torch.cuda.set_device(opt.cGPU)
    device = torch.device("cuda")
    print('device: {} {}'.format(device, torch.cuda.current_device()))

    # build model and load parameters
    reparsed_network = '{}.{}.networks.{}'.format(opt.resume.split('/')[-2], opt.resume.split('/')[-1], opt.net)
    network = importlib.import_module(reparsed_network)
    model = network.get_model(opt, require_init=False)
    resume_ckpt = os.path.join(opt.resume, 'checkpoint_best.pth.tar')
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    opt.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    # load dataset
    if opt.split == 'val':
        whole_test_set = dataset.get_whole_val_set(opt)
    elif opt.split == 'test':
        whole_test_set = dataset.get_whole_test_set(opt)
    print('database:{}, query:{}'.format(whole_test_set.dbStruct.numDb, whole_test_set.dbStruct.numQ))

    # evaluate
    recalls = evaluate.get_recall(opt, model, whole_test_set, seed_worker)

    # export results
    with open(os.path.join(opt.runsPath, 'evaluate.log'), 'a') as f:
        f.write('[{}]\t'.format(opt.split))
        f.write('recall@1: {:.2f}\t'.format(recalls[1]))
        f.write('recall@5: {:.2f}\t'.format(recalls[5]))
        f.write('recall@10: {:.2f}\t'.format(recalls[10]))
        f.write('recall@20: {:.2f}\n'.format(recalls[20]))
        f.flush()
    return recalls


def train(opt, seed_worker=None, trial=None):
    # --------------------------------------- 1. set device -------------------------------------- #
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    torch.cuda.set_device(opt.cGPU)
    device = torch.device("cuda")
    print('train.py device: {} {}'.format(device, torch.cuda.current_device()))
    # ---------------------------------------- 2A. resume ---------------------------------------- #
    if opt.resume != '':
        # load model
        print('resume path:', opt.resume)
        opt = update_opt_from_json(os.path.join(opt.resume, 'flags.json'), opt)
        opt.runsPath = opt.resume

        reparsed_network = '{}.{}.networks.network_{}'.format(opt.resume.split('/')[-2], opt.resume.split('/')[-1], opt.net)
        network = importlib.import_module(reparsed_network)
        model = network.get_model(opt, require_init=False)
        resume_ckpt = os.path.join(opt.resume, 'checkpoint_last.pth.tar')
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

        # load optimizer
        if opt.optim == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weightDecay)
            if not opt.train_att:
                optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        elif opt.optim == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)

        opt.start_epoch = checkpoint['epoch']
        print('current epoch:', opt.start_epoch)

        if opt.train_att:
            model = network.freeze_layers(opt, model)

    # ---------------------------------- 2B. create new training --------------------------------- #
    else:
        with open(os.path.join(opt.structDir, 'pcl_parameter.json'), 'r') as f:
            record = json.load(f)
        opt.runsPath = os.path.join(opt.logsPath, opt.imgDir.split('/')[-2] + '_' + opt.net + '_' + '_seq' + str(opt.seqLen) + '_' + opt.comment + '_' + datetime.now().strftime('%m%d%H%M%S%f'))

        if not os.path.exists(opt.logsPath):
            os.mkdir(opt.logsPath)

        if not os.path.exists(opt.runsPath):
            os.mkdir(opt.runsPath)
            os.mkdir(os.path.join(opt.runsPath, 'networks'))

        # build model
        assert os.path.exists('networks/{}.py'.format(opt.net)), 'cannot find ' + 'network_{}.py'.format(opt.net)
        network = importlib.import_module('networks.' + opt.net)
        for file in [__file__, 'nuscene.py', 'networks/{}.py'.format(opt.net)]:
            shutil.copyfile(file, os.path.join(opt.runsPath, 'networks', file.split('/')[-1]))

        model = network.get_model(opt, require_init=False)                      # summary(model, input_size=(3, 3, 200, 200), batch_size=32)
        print('current model:=================\n')
        model_state_dict = model.state_dict()
        for k in model_state_dict.keys():
            print(k)
        print('\n')

        checkpoint = torch.load('vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar', map_location=lambda storage, loc: storage)
        print('pretrained model:=================\n')
        pretrained_model_dict = checkpoint['state_dict']
        pretrained_model_dict['encoder.encoder.1.weight'] = pretrained_model_dict.pop('encoder.0.weight')
        pretrained_model_dict['encoder.encoder.1.bias'] = pretrained_model_dict.pop('encoder.0.bias')
        pretrained_model_dict['encoder.encoder.5.weight'] = pretrained_model_dict.pop('encoder.2.weight')
        pretrained_model_dict['encoder.encoder.5.bias'] = pretrained_model_dict.pop('encoder.2.bias')
        pretrained_model_dict['encoder.encoder.11.weight'] = pretrained_model_dict.pop('encoder.5.weight')
        pretrained_model_dict['encoder.encoder.11.bias'] = pretrained_model_dict.pop('encoder.5.bias')
        pretrained_model_dict['encoder.encoder.15.weight'] = pretrained_model_dict.pop('encoder.7.weight')
        pretrained_model_dict['encoder.encoder.15.bias'] = pretrained_model_dict.pop('encoder.7.bias')
        pretrained_model_dict['encoder.encoder.21.weight'] = pretrained_model_dict.pop('encoder.10.weight')
        pretrained_model_dict['encoder.encoder.21.bias'] = pretrained_model_dict.pop('encoder.10.bias')
        pretrained_model_dict['encoder.encoder.25.weight'] = pretrained_model_dict.pop('encoder.12.weight')
        pretrained_model_dict['encoder.encoder.25.bias'] = pretrained_model_dict.pop('encoder.12.bias')
        pretrained_model_dict['encoder.encoder.29.weight'] = pretrained_model_dict.pop('encoder.14.weight')
        pretrained_model_dict['encoder.encoder.29.bias'] = pretrained_model_dict.pop('encoder.14.bias')
        pretrained_model_dict['encoder.encoder.35.weight'] = pretrained_model_dict.pop('encoder.17.weight')
        pretrained_model_dict['encoder.encoder.35.bias'] = pretrained_model_dict.pop('encoder.17.bias')
        pretrained_model_dict['encoder.encoder.39.weight'] = pretrained_model_dict.pop('encoder.19.weight')
        pretrained_model_dict['encoder.encoder.39.bias'] = pretrained_model_dict.pop('encoder.19.bias')
        pretrained_model_dict['encoder.encoder.43.weight'] = pretrained_model_dict.pop('encoder.21.weight')
        pretrained_model_dict['encoder.encoder.43.bias'] = pretrained_model_dict.pop('encoder.21.bias')
        pretrained_model_dict['encoder.encoder.49.weight'] = pretrained_model_dict.pop('encoder.24.weight')
        pretrained_model_dict['encoder.encoder.49.bias'] = pretrained_model_dict.pop('encoder.24.bias')
        pretrained_model_dict['encoder.encoder.53.weight'] = pretrained_model_dict.pop('encoder.26.weight')
        pretrained_model_dict['encoder.encoder.53.bias'] = pretrained_model_dict.pop('encoder.26.bias')
        pretrained_model_dict['encoder.encoder.57.weight'] = pretrained_model_dict.pop('encoder.28.weight')
        pretrained_model_dict['encoder.encoder.57.bias'] = pretrained_model_dict.pop('encoder.28.bias')
        for k in pretrained_model_dict.keys():
            print(k)
        print('\n')

        model.load_state_dict(pretrained_model_dict, strict=False)

        # build optimizer
        if opt.optim == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weightDecay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        elif opt.optim == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)

    def unexpected_exit():
        if opt.resume == '':
            import shutil
            shutil.rmtree(opt.runsPath)
            print('unexpected exit: remove current log.')
        else:
            print('resume stops')
    atexit.register(unexpected_exit)

    model = model.to(device)
    if opt.nGPU > 1:
        model = nn.DataParallel(model)

    # ------------------------------------- 3. loss function ------------------------------------- #
    criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, p=2, reduction='sum').to(device)

    # -------------------------------------- 4. load dataset ------------------------------------- #
    # for feature cache
    whole_train_set = dataset.get_whole_training_set(opt)
    whole_training_data_loader = DataLoader(dataset=whole_train_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=cuda, worker_init_fn=seed_worker)
    whole_val_set = dataset.get_whole_val_set(opt)
    whole_val_data_loader = DataLoader(dataset=whole_val_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=cuda, worker_init_fn=seed_worker)
    whole_test_set = dataset.get_whole_test_set(opt)
    # for train tuples
    train_set = dataset.get_training_query_set(opt, opt.margin)
    val_set = dataset.get_val_query_set(opt, opt.margin)
    print('train database:{}, training query:{}, val query:{}, test query:{}'.format(train_set.dbStruct.numDb, len(train_set), whole_val_set.dbStruct.numQ, whole_test_set.dbStruct.numQ))

    # -------------------------------------- 5. tensorboard -------------------------------------- #
    writer = SummaryWriter(log_dir=opt.runsPath)
    with open(os.path.join(opt.runsPath, 'flags.json'), 'w') as f:
        f.write(json.dumps({k: v for k, v in vars(opt).items()}, indent=''))

    # ---------------------------------------- 6. training --------------------------------------- #
    not_improved = 0
    best_recall_at_1 = 0
    for epoch in range(opt.start_epoch + 1, opt.nEpochs + 1):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        epoch_loss = 0
        startIter = 1  # keep track of batch iter across subsets for logging

        nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

        # ------------------------------------ 6.1 build cache ----------------------------------- #
        print('build cache..')
        model.eval()
        train_set.cache = os.path.join(opt.runsPath, train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5:
            h5feat = h5.create_dataset("features", [len(whole_train_set), opt.output_dim], dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(tqdm.tqdm(whole_training_data_loader, ncols=40), 1):
                    input = input.to(device)                                   # torch.Size([32, 3, 154, 154]) ([32, 5, 3, 200, 200])
                    # with ef.scan(enabled=False):
                    vlad_encoding = model(input)
                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    del input, vlad_encoding

        # ------------------------------------- 6.2 training ------------------------------------- #
        print('training..')
        model.train()
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, collate_fn=dataset.collate_fn, pin_memory=cuda, worker_init_fn=seed_worker)  # pin_memory=cuda ?
        for iteration, (query, positives, negatives, negCounts, indices) in enumerate(tqdm.tqdm(training_data_loader, ncols=40), startIter):
            if query is None:
                continue  # in case we get an empty batch

            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor, where N = batchSize * (nQuery + nPos + nNeg)
            B, L, C, H, W = query.shape                                        # ([8, 3, 200, 200])
            nNeg = torch.sum(negCounts)                                        # tensor(80) = torch.sum(torch.Size([8]))
            input = torch.cat([query, positives, negatives])                   # ([96, 3, 200, 200]) = torch.cat(([8, 3, 200, 200]), ([8, 3, 200, 200]), ([80, 3, 200, 200]), ([8]))

            # input device: cpu, # input device: cuda 1, so what is the point of pin_memory?
            input = input.to(device)                                           # ([96, 1, 3, 200, 200])
            vlad_encoding = model(input)
            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

            optimizer.zero_grad()

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(vladQ[i:i + 1], vladP[i:i + 1], vladN[negIx:negIx + 1])

            loss /= nNeg.float().to(device)  # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del input, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            # --------- clipping to limit the magnitude of the backpropagated gradients to a value of 80 --------- #
            torch.nn.utils.clip_grad_norm_(model.parameters(), 80, norm_type=2.0)   # [3/3]kid

            if iteration % 10 == 0 or nBatches <= 10 or iteration == 1:
                writer.add_scalar('train_batch_loss', batch_loss, ((epoch - 1) * nBatches) + iteration)
                writer.add_scalar('train_batch_nNeg', nNeg, ((epoch - 1) * nBatches) + iteration)

        startIter += len(training_data_loader)
        del training_data_loader
        if 'loss' in locals():
            del loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        os.remove(train_set.cache)  # delete HDF5 cache

        train_avg_loss = epoch_loss / nBatches
        writer.add_scalar('train_epoch_avg_loss', train_avg_loss, epoch)
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(name + '_grad', param.grad, epoch)
                writer.add_histogram(name + '_data', param, epoch)

        if opt.optim == 'sgd':
            scheduler.step()

        if (epoch % opt.evalEvery) == 0:
            current_recalls = evaluate.get_recall(opt, model, whole_val_set, seed_worker, epoch, writer)
            is_best = 0

            if epoch > 40:
                is_best = current_recalls[1] > best_recall_at_1
                if is_best:
                    not_improved = 0
                    best_recall_at_1 = current_recalls[1]
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'recalls': current_recalls,
                        'best_recall_at_1': best_recall_at_1,
                        'optimizer': optimizer.state_dict()
                    }, os.path.join(opt.runsPath, 'checkpoint_best.pth.tar'))
                else:
                    not_improved += 1

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

            quick_log('screen.log', 'epoch: {:>2d}\t'.format(epoch), 'lr: {:>.8f}\t'.format(current_lr), 'train loss: {:>.4f}\t'.format(train_avg_loss),
                      'recall@1: {:.2f}\t'.format(current_recalls[1]), 'recall@5: {:.2f}\t'.format(current_recalls[5]), 'recall@10: {:.2f}\t'.format(current_recalls[10]),
                      'recall@20: {:.2f}\t'.format(current_recalls[20]), '*\n' if is_best else '\n')

    writer.close()
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'recalls': current_recalls,
        'best_recall_at_1': best_recall_at_1,
        'optimizer': optimizer.state_dict()
    }, os.path.join(opt.runsPath, 'checkpoint_last.pth.tar'))
    atexit.unregister(unexpected_exit)
    return current_recalls[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch-NetVlad')
    parser.add_argument('--structDir', type=str, default='dataset/7n5s_xy11', help='Path for structure.')
    parser.add_argument('--imgDir', type=str, default='dataset/7n5s_xy11/img_polar', help='Path for images.')
    parser.add_argument('--comment', type=str, default='', help='comment')
    parser.add_argument('--seqLen', type=int, default=1, help='number of sequence to use.')
    parser.add_argument('--mode', type=str, default='train', help='mode', choices=['train', 'evaluate', 'hyper'])
    parser.add_argument('--net', type=str, default='kid', help='network')
    parser.add_argument('--batchSize', type=int, default=8, help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
    parser.add_argument('--cacheBatchSize', type=int, default=32, help='Batch size for caching and testing')
    parser.add_argument('--cacheRefreshRate', type=int, default=0, help='How often to refresh cache, in number of queries. 0 for off')
    parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
    parser.add_argument('--cGPU', type=int, default=1, help='core of GPU to use.')  # modified
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer to use', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate.')   # [1/3]kid
    parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
    parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
    parser.add_argument('--weightDecay', type=float, default=1e-7, help='Weight decay for SGD.')            # [2/3]kid
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
    parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
    parser.add_argument('--logsPath', type=str, default='./logs_kid', help='Path to save runs to.')
    parser.add_argument('--runsPath', type=str, default='./eval', help='Path to save runs to.')
    parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
    parser.add_argument('--evalEvery', type=int, default=1, help='Do a validation set run, and save, every N epochs.')
    parser.add_argument('--patience', type=int, default=0, help='Patience for early stopping. 0 is off.')
    parser.add_argument('--split', type=str, default='val', help='Split to use', choices=['val', 'test'])
    parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
    parser.add_argument('--encoder_dim', type=int, default=512, help='Number of feature dimension. Default=512')
    parser.add_argument('--output_dim', type=int, default=32768, help='Number of feature dimension. Default=512')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss. Default=0.1')
    parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
    opt = parser.parse_args()

    # fix_random = FixRandom(opt.seed)
    # seed_worker = fix_random.seed_worker
    # fix_random.set_everything_fixed()

    if opt.mode == 'train':
        last_recall_1 = train(opt)
        print('last_recall_1:', last_recall_1)
    elif opt.mode == 'evaluate':
        evaluate_model(opt)
