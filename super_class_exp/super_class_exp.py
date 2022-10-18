import argparse
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import wandb
from scipy.stats import sem
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models import ZSCombiner, ZSCombiner_2l, HNForResnet, WWHN
from utils import Logger, load_object

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args():
    parser = argparse.ArgumentParser(description='Create model from domain description')
    parser.add_argument("--wandb_project", type=str, default='D3A')
    parser.add_argument("--wandb_user", type=str, default='tomervolk')
    parser.add_argument("--log_file", type=str, default='')
    parser.add_argument("--run_name", type=str, default='organizing')

    #################################
    #       Datasets Parameters     #
    #################################
    parser.add_argument("--path", type=str, default='./')
    parser.add_argument("--resize", default=224, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=20, type=int)
    # parser.add_argument("--remove_fc", action='store_true')

    #################################
    #       Hypernet Parameters     #
    #################################
    parser.add_argument("--hn_type", type=str, default='W', choices=['WW', 'W'])
    parser.add_argument("--hnet_hidden_size", default=120, type=int)

    #################################
    #       Training Parameters     #
    #################################
    parser.add_argument("--hn_train_epochs", default=20, type=int)
    parser.add_argument("--inner_train_epochs", default=2, type=int)
    parser.add_argument("--inner_lr", default=0.01, type=float)
    parser.add_argument("--inner_momentum", default=0.9, type=float)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)

    args = parser.parse_args()
    return args


class mySBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2').cuda()

    def forward(self, cname):
        embeddings = self.sbert.encode([cname.replace('not-', '')])
        return torch.tensor(embeddings, device='cuda')


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_encoder = mySBERT().cuda()
        self.out_dim = 384

    def forward(self, descriptor):
        e1 = self.text_encoder.forward(descriptor)
        # return torch.cat([e1, e2], dim=1)
        return [e1]


@torch.no_grad()
def eval_hresnet(hnet, text_encoder, combiner, data, logger, pretext='', metrics={}, device='cuda'):
    accs = []
    hnet.eval()
    # text_encoder.eval()
    ce = torch.nn.BCEWithLogitsLoss()
    descriptors_accs = dict()
    running_correct, running_loss, n = 0., 0., 0.

    for descriptor in data[pretext]['descriptions']:
        l0, l1 = descriptor.split(' ')
        c_class2idx = data[pretext]['class2idx']
        X0 = data[pretext]['X'][c_class2idx[l0]]
        X1 = data[pretext]['X'][c_class2idx[l1]]
        X = torch.cat([X0, X1])
        y = torch.tensor([0] * X0.shape[0] + [1] * X1.shape[0])
        perm_ind = torch.randperm(y.shape[0])

        descriptor_embed = text_encoder.forward(l0)
        descriptor_embed = [z.detach().float() for z in descriptor_embed]
        weights = hnet(descriptor_embed)
        combiner.load_state_dict(weights)
        combiner.eval()

        j = 0
        batch_features, labels = [], []
        while j * args.batch_size < perm_ind.shape[0]:
            batch_ind = perm_ind[j * args.batch_size:(j + 1) * args.batch_size]
            Xb, yb = X[batch_ind], y[batch_ind]
            Xb, yb = Xb.to('cuda'), yb.to('cuda')
            j += 1

            pred = combiner(Xb).squeeze(1)
            loss = ce(pred, yb.float())
            running_loss += loss.item()
            batch_features.append(pred)
            labels.append(yb)
            # running_correct += pred.argmax(1).eq(yb).sum().item()
            n += 1

        # acc = running_correct / n
        auc = roc_auc_score(y_true=torch.cat(labels).tolist(), y_score=torch.cat(batch_features).tolist())
        accs.append(auc)
        # logger.print_and_log('Descriptor: %s, Acc: %s ' % (descriptor, acc), just_log=True)
        descriptors_accs['hnet_prediction_on_%s_%s' % (descriptor, pretext)] = auc

    logger.print_and_log(pretext + ' Average auc: %s' % (np.mean(accs)))

    metrics['hnet_auc_%s' % pretext] = np.mean(accs)
    metrics['hnet_median_%s' % pretext] = np.median(accs)
    metrics['hnet_std_%s' % pretext] = np.std(accs)
    metrics['hnet_SEM_%s' % pretext] = sem(accs)

    return metrics, descriptors_accs


def get_text_encoder():
    text_encoder = TextEncoder()
    text_encoder_out_dim = text_encoder.out_dim

    return text_encoder.cuda(), text_encoder_out_dim


def get_data():
    data = {'train_visual_feature_extractor': dict(),
            # 'eval train': dict(),
            'dev': dict(),
            'test seen': dict(),
            'train_hn': dict(),
            'zs_classes': dict()
            }

    dataset_folder = 'awa2'
    visual_features_dim = 512
    for k in data.keys():
        print(k)
        (data[k]['X'], data[k]['class2idx']) = load_object('%s/%s.pkl' % (dataset_folder, k))
        data[k]['labels'] = [x for x in data[k]['class2idx'] if not x.startswith('not-')]
        data[k]['descriptions'] = \
            ['%s %s' % ('not-' + l1, l1) for l1 in data[k]['labels']]
        pass
    return data, visual_features_dim


def get_hnet(text_encoder_out_dim, resnet_out_dim, hn_hidden_dim, out_dim, hn_type):
    if hn_type == 'W':
        hnet = HNForResnet(out_dim * text_encoder_out_dim, resnet_out_dim, hn_hidden_dim=hn_hidden_dim,
                           out_dim=out_dim).cuda()
        return hnet
    elif hn_type == 'WW':
        return WWHN(out_dim * text_encoder_out_dim, resnet_out_dim, hn_hidden_dim=hn_hidden_dim,
                    target_hidden_dim=hn_hidden_dim, target_out_dim=out_dim).cuda()
    else:
        print('Error! hn_type was not found!')
        exit()


def get_combiner(input_dim, hidden_dim, hn_type):
    if hn_type == 'W':
        return ZSCombiner(input_dim=input_dim, out_dim=1).cuda()
    elif hn_type == 'WW':
        return ZSCombiner_2l(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=1).cuda()
    else:
        print('Error! hn_type combiner was not found!')
        exit()


if __name__ == '__main__':
    args = get_args()
    logger = Logger(filename='%s' % args.log_file)
    logger.log_object(args, 'args')
    # wandb.init(config=args, entity=args.wandb_user,
    #            project='%s_%s_%s' % (args.wandb_project, args.dataset, args.text_encoder),
    #            settings=wandb.Settings(start_method='fork'))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, visual_features_dim = get_data()
    text_encoder, text_encoder_out_dim = get_text_encoder()

    hnet = get_hnet(text_encoder_out_dim, visual_features_dim, hn_hidden_dim=args.hnet_hidden_size,
                    out_dim=1, hn_type=args.hn_type)
    combiner = get_combiner(input_dim=visual_features_dim, hidden_dim=args.hnet_hidden_size,
                            hn_type=args.hn_type)

    metrics = dict()
    hnet_optimizer = torch.optim.SGD(hnet.parameters(), lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay)
    ce = torch.nn.BCEWithLogitsLoss()

    metrics['best_val_auc'], metrics['best_test_auc'], metrics['best_test_auc_std'], metrics[
        'best_test_auc_SEM'] = 0.0, 0.0, 0.0, 0.0
    for epoch in tqdm(range(args.hn_train_epochs)):
        hnet.train()

        all_descriptors = data['train_visual_feature_extractor']['descriptions'] + \
                          data['train_hn']['descriptions']
        random.shuffle(all_descriptors)
        for descriptor in all_descriptors[:100]:
            # loader = descriptor2loader[descriptor]
            # get data
            l0, l1 = descriptor.split(' ')
            if l0.replace('not-', '') in data['train_visual_feature_extractor']['class2idx']:
                c_class2idx = data['train_visual_feature_extractor']['class2idx']
                X0 = data['train_visual_feature_extractor']['X'][c_class2idx[l0]]
                X1 = data['train_visual_feature_extractor']['X'][c_class2idx[l1]]
            else:
                c_class2idx = data['train_hn']['class2idx']
                X0 = data['train_hn']['X'][c_class2idx[l0]]
                X1 = data['train_hn']['X'][c_class2idx[l1]]

            if len(X0) > len(X1):
                perm_ind = torch.randperm(X0.shape[0])
                X0 = X0[perm_ind]
                X0 = X0[:len(X1)]
            else:
                perm_ind = torch.randperm(X1.shape[0])
                X1 = X1[perm_ind]
                X1 = X1[:len(X0)]

            X = torch.cat([X0, X1])
            if 'not-' in l0:
                y = torch.tensor([0] * X0.shape[0] + [1] * X1.shape[0])
            else:
                y = torch.tensor([1] * X0.shape[0] + [0] * X1.shape[0])
            perm_ind = torch.randperm(y.shape[0])

            descriptor_embed = text_encoder.forward(l0)
            descriptor_embed = [z.detach().float() for z in descriptor_embed]
            weights = hnet(descriptor_embed)

            combiner.load_state_dict(weights)
            inner_optim = torch.optim.SGD(combiner.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

            for i in range(args.inner_train_epochs):
                running_correct, running_loss, n = 0., 0., 0.
                losses = []

                combiner.train()

                j = 0
                batch_features, labels = [], []
                while j * args.batch_size < perm_ind.shape[0]:
                    batch_ind = perm_ind[j * args.batch_size:(j + 1) * args.batch_size]
                    Xb, yb = X[batch_ind], y[batch_ind]
                    Xb, yb = Xb.to('cuda'), yb.to('cuda')
                    j += 1

                    pred = combiner(Xb).squeeze(1)
                    loss = ce(pred, yb.float())
                    inner_optim.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(combiner.parameters(), 50)
                    inner_optim.step()

                    running_loss += loss.item()
                    losses.append(loss.item())
                    batch_features += pred.clone().detach().tolist()
                    labels += yb.clone().detach().tolist()
                    # running_correct += pred.argmax(1).eq(yb).sum().item()
                    n += 1
                cur_auc = roc_auc_score(labels, batch_features)
                print('%s Iteration: %d, Train loss: %s, Train AUC %s' % (descriptor, i, running_loss / n, cur_auc))
                wandb.log({'Average train loss': (running_loss / n)})

            hnet_optimizer.zero_grad()
            grad_params = [p for p in hnet.parameters() if p.requires_grad]

            final_state = combiner.state_dict()
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

            grads = torch.autograd.grad(
                list(weights.values()), grad_params, grad_outputs=list(delta_theta.values()), allow_unused=True)

            # update hnet weights
            for p, g in zip(grad_params, grads):
                p.grad = g
            hnet_optimizer.step()

        descriptors_accs = dict()
        for split in ['train_visual_feature_extractor', 'train_hn', 'dev',
                      'test seen', 'zs_classes']:
            metrics, descriptors_accs[split] = eval_hresnet(hnet, text_encoder, combiner, data, logger, pretext=split,
                                                            metrics=metrics)
        sys.stdout.flush()
        wandb.log(metrics)
    logger.log_object(descriptors_accs, 'descriptors_aucs')

    wandb.finish()
