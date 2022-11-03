import numpy as np
from scipy.stats import sem
from tqdm import tqdm
from dataset import get_hn_loaders
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ZSCombiner, ZSCombiner_2l, HNForResnet, AOClevrNet, EVHN, IVEVHN, WWHN
from collections import OrderedDict
import os
import sys
import argparse
from utils import Logger, load_object
import time
import random
# from transformers import BertTokenizer, BertModel
from torchvision.models import resnet18
# from torchvision.models import ResNet18_Weights
from sentence_transformers import SentenceTransformer
from point_clouds_models import PointNetCls
import itertools
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args():
    parser = argparse.ArgumentParser(description='Create model from domain description')
    parser.add_argument("--log_file", type=str, default='')

    #################################
    #       Datasets Parameters     #
    #################################
    parser.add_argument("--path", type=str, default='/mnt/dsi_vol1/users/amosy/data/Animals_with_Attributes2/JPEGImages/') #[/cortex/data/images/Animals_with_Attributes2/JPEGImages/, /mnt/dsi_vol1/users/amosy/data/Animals_with_Attributes2/JPEGImages/]
    parser.add_argument("--dataset", type=str, default='AwA', choices=['AO_clevr', 'AwA', 'PC', 'CUB', 'SUN'])
    parser.add_argument("--resize", default=224, type=int)
    parser.add_argument("--val_frac", default=0.3, type=float)
    parser.add_argument("--out_dim", default=2, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=20, type=int)
    parser.add_argument("--nattributes", default=4, type=int)
    parser.add_argument("--use_attributes", action='store_true')
    # parser.add_argument("--remove_fc", action='store_true')

    #################################
    #       Hypernet Parameters     #
    #################################
    parser.add_argument("--hn_type", type=str, default='EV', choices=['IVEV', 'EV', 'WW', 'W'])
    # parser.add_argument("--pretrain_resnet", default='pretrain/pretrain_resnet18_50multilabels.pt', type=str)
    parser.add_argument("--tuned_resnet", default=0.0, type=float)
    parser.add_argument("--embedding_dim", default=512, type=int)
    parser.add_argument("--text_encoder", default='w2v', type=str, choices=['SBERT', 'w2v', 'TFIDF'])
    parser.add_argument("--encode_text_as", default='dense', type=str, choices=['similarity', 'dense'])
    parser.add_argument("--n_hidden", default=1, type=int)
    parser.add_argument("--hnet_hidden_size", default=120, type=int)
    parser.add_argument("--p_negative", default=0.0, type=float)

    #################################
    #       Training Parameters     #
    #################################
    parser.add_argument("--hn_train_epochs", default=20, type=int)
    parser.add_argument("--inner_train_epochs", default=2, type=int)
    parser.add_argument("--inner_lr", default=0.01, type=float)
    parser.add_argument("--inner_momentum", default=0.9, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--optimizer_type", default='sgd', type=str, choices=['sgd'])


    # Text encoder
    parser.add_argument("--tune_text_encoder_until", default=1, type=int)
    parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
    parser.add_argument("--text_encoder_momentum", default=0.001, type=float)
    parser.add_argument("--text_encoder_weight_decay", default=1e-5, type=float)
    parser.add_argument("--text_encoder_adamw_eps", default=1e-8, type=float)


    args = parser.parse_args()
    return args


def padd_batch(batch, max_len=512):
    return [z + [0] * (max_len - len(z)) for z in batch]


class myBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased').to('cuda')

    def forward(self, cname):
        text_list = ['[CLS]', cname] if len(cname.split('+')) == 2 else ['[CLS]'] + cname.split('+')
        tokenized_data = self.tokenizer(text_list, is_split_into_words=True, truncation=True, max_length=10,
                                                return_tensors="pt", return_attention_mask=True)
        bert_descriptors = self.bert(input_ids=tokenized_data["input_ids"].cuda(),
                                             attention_mask=tokenized_data["attention_mask"].cuda())['pooler_output']
        return bert_descriptors


class myW2V():
    def __init__(self):
        self.class2embedding = load_object('labels_w2v_300.pkl')

    def forward(self, text):
        return self.class2embedding[text].unsqueeze(0).cuda()


class mySBERT(nn.Module):
    def __init__(self, dataset='AwA'):
        super().__init__()
        self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2').cuda()
        if dataset == 'AwA':
            self.class2descriptors_list = load_object('gpt_label2descriptors.pkl')
        elif dataset == 'CUB':
            self.class2descriptors_list = load_object('cub/label2descriptions.pkl')
        self.dataset = dataset

    def forward(self, cname):
        if self.dataset == 'AwA':
            descriptors = self.class2descriptors_list[cname.replace('+', ' ')]
            random_descriptor = random.choice(descriptors)
            embeddings = self.sbert.encode([random_descriptor])

        elif self.dataset == 'CUB':
            descriptors = self.class2descriptors_list[cname.split('.')[0]]
            random_descriptor = random.choice(descriptors)
            embeddings = self.sbert.encode([random_descriptor])
        elif self.dataset == 'SUN':
            descriptors = ' '.join(cname.split('/')[2:]).replace('_', ' ')
            embeddings = self.sbert.encode([descriptors])

        elif self.dataset in ('AO_clevr', 'PC'):
            embeddings = self.sbert.encode([cname])

        return torch.tensor(embeddings, device='cuda')


class myNSBERT(nn.Module):
    def __init__(self, p_negatives=0.0, nattributes=4):
        super().__init__()
        self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2').cuda()
        self.class2descriptors_list = load_object('gpt_label2descriptors.pkl')
        self.label2attributes_names = load_object('label2attributes_names.pkl')
        self.p_negatives = p_negatives
        self.nattributes = nattributes

    def get_negative_attributes(self, l1, l2, nattributes):
        s1 = set(self.label2attributes_names[l1])
        s2 = set(self.label2attributes_names[l2])

        not_l1 = list(s2 - s1)
        not_l2 = list(s1 - s2)

        random.shuffle(not_l1)
        random.shuffle(not_l2)

        d1 = 'Animal that is not: ' + ' '.join(not_l1[:nattributes])
        d2 = 'Animal that is not: ' + ' '.join(not_l2[:nattributes])
        return d1, d2

    def forward(self, descriptor):
        l1, l2 = descriptor.split(' ')
        if np.random.rand() > self.p_negatives:
            descriptors1 = self.class2descriptors_list[l1.replace('+', ' ')]
            descriptors2 = self.class2descriptors_list[l2.replace('+', ' ')]
            embeddings = self.sbert.encode([random.choice(descriptors1), random.choice(descriptors2)])

        else:
            descriptors1, descriptors2 = self.get_negative_attributes(l1, l2, self.nattributes)
            embeddings = self.sbert.encode([descriptors1, descriptors2])
        embeddings = [torch.tensor(z, device='cuda').unsqueeze(0) for z in embeddings]
        return embeddings


class myTFIDF(nn.Module):
    def __init__(self, dataset, p_negatives=0.0, nattributes=4):
        super().__init__()
        if dataset == 'AwA':
            self.class2descriptors_list = load_object('gpt_label2descriptors.pkl')
            self.label2attributes_names = load_object('label2attributes_names.pkl')
            self.vectorizer = load_object('awa/tfidf.pkl')
            self.out_dim = 686
        elif dataset == 'CUB':
            self.class2descriptors_list = load_object('cub/label2descriptors.pkl')
            self.vectorizer = load_object('cub/tfidf.pkl')
            self.out_dim = 4255
        else:
            print('Error! Dataset was not found!')
            exit()

        self.p_negatives = p_negatives
        self.nattributes = nattributes
        self.dataset = dataset

    def get_negative_attributes(self, l1, l2, nattributes):
        s1 = set(self.label2attributes_names[l1])
        s2 = set(self.label2attributes_names[l2])

        not_l1 = list(s2 - s1)
        not_l2 = list(s1 - s2)

        random.shuffle(not_l1)
        random.shuffle(not_l2)

        d1 = 'Animal that is not: ' + ' '.join(not_l1[:nattributes])
        d2 = 'Animal that is not: ' + ' '.join(not_l2[:nattributes])
        return d1, d2

    def forward(self, descriptor):
        l1, l2 = descriptor.split(' ')
        if np.random.rand() > self.p_negatives:
            descriptors1 = self.class2descriptors_list[l1.replace('+', ' ')]
            descriptors2 = self.class2descriptors_list[l2.replace('+', ' ')]
            embeddings = self.vectorizer.transform([random.choice(descriptors1), random.choice(descriptors2)]).toarray()

        else:
            descriptors1, descriptors2 = self.get_negative_attributes(l1, l2, self.nattributes)
            embeddings = self.vectorizer.transform([descriptors1, descriptors2]).toarray()
        embeddings = [torch.tensor(z, device='cuda').unsqueeze(0) for z in embeddings]
        return embeddings


class Similarity(nn.Module):
    def __init__(self, span_labels, encoder='BERT', dataset='AwA'):
        super().__init__()
        self.span_labels = span_labels
        self.encoder = encoder
        self.span_base = []
        self.cs = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.dataset = dataset

        if self.encoder == 'w2v':
            self.text_encoder = myW2V()

        elif self.encoder == 'BERT':
            self.text_encoder = myBERT().cuda()

        elif self.encoder == 'SBERT':
            self.text_encoder = mySBERT(self.dataset ).cuda()

    def forward(self, descriptor):
        self.span_base = []
        for l in self.span_labels:
            self.span_base.append(self.text_encoder.forward(l))

        if self.dataset == 'AwA':
            d, l1, l2 = descriptor.split(' ')
        elif self.dataset == 'AO_clevr':
            d1, d2, d3, d4 = descriptor.split(' ')
            l1 = '%s %s' % (d1, d2)
            l2 = '%s %s' % (d3, d4)
        elif self.dataset in ('PC', 'CUB'):
            l1, l2 = descriptor.split(' ')

        e1 = self.text_encoder.forward(l1)
        e2 = self.text_encoder.forward(l2)

        z1 = torch.zeros(len(self.span_labels) * 2)
        for i, el in enumerate(self.span_base):
            z1[i] = self.cs(e1, el)
            z1[i + len(self.span_labels)] = self.cs(e2, el)

        return z1.to('cuda')


class TextEncoder(nn.Module):
    def __init__(self, text_encoder_type, dataset='AwA'):
        super().__init__()
        self.dataset = dataset
        if text_encoder_type == 'w2v':
            self.text_encoder = myW2V()
            self.out_dim = 300

        elif text_encoder_type == 'SBERT':
            self.text_encoder = mySBERT(self.dataset).cuda()
            self.out_dim = 384

    def forward(self, descriptor):
        if self.dataset == 'AwA':
            l1, l2 = descriptor.split(' ')
        elif self.dataset == 'AO_clevr':
            d1, d2, d3, d4 = descriptor.split(' ')
            l1 = '%s %s' % (d1, d2)
            l2 = '%s %s' % (d3, d4)
        elif self.dataset in ('PC', 'CUB', 'SUN'):
            l1, l2 = descriptor.split(' ')
        else:
            print('Dataset was not detected!')
        e1 = self.text_encoder.forward(l1)
        e2 = self.text_encoder.forward(l2)
        # return torch.cat([e1, e2], dim=1)
        return [e1, e2]


@torch.no_grad()
def eval_hresnet(hnet, text_encoder, combiner, data, logger, pretext='', metrics={}, device='cuda'):

    accs = []
    hnet.eval()
    # text_encoder.eval()
    ce = torch.nn.CrossEntropyLoss()
    descriptors_accs = dict()
    running_correct, running_loss, n = 0., 0., 0.

    for descriptor in data[split]['descriptions']:
        # test_loader = descriptor2loader[descriptor]
        l0, l1 = descriptor.split(' ')
        if l0 in data['eval_visual_feature_extractor']['class2idx']:
            c_class2idx = data['eval_visual_feature_extractor']['class2idx']
            X0 = data['eval_visual_feature_extractor']['X'][c_class2idx[l0]]
            X1 = data['eval_visual_feature_extractor']['X'][c_class2idx[l1]]
        elif l0 in data['eval_hn']['class2idx']:
            c_class2idx = data['eval_hn']['class2idx']
            X0 = data['eval_hn']['X'][c_class2idx[l0]]
            X1 = data['eval_hn']['X'][c_class2idx[l1]]
        elif l0 in data['zs_classes']['class2idx']:
            c_class2idx = data['zs_classes']['class2idx']
            X0 = data['zs_classes']['X'][c_class2idx[l0]]
            X1 = data['zs_classes']['X'][c_class2idx[l1]]
        else:
            print('Warninig!! labels does not found!')
        X = torch.cat([X0, X1])
        y = torch.tensor([0] * X0.shape[0] + [1] * X1.shape[0])
        perm_ind = torch.randperm(y.shape[0])

        descriptor_embed = text_encoder.forward(descriptor)
        descriptor_embed = [z.detach().float() for z in descriptor_embed]
        weights = hnet(descriptor_embed)
        combiner.load_state_dict(weights)
        combiner.eval()

        j = 0
        batch_features = []
        while j * args.batch_size < perm_ind.shape[0]:
            batch_ind = perm_ind[j * args.batch_size:(j + 1) * args.batch_size]
            Xb, yb = X[batch_ind], y[batch_ind]
            Xb, yb = Xb.to('cuda'), yb.to('cuda')
            j += 1

            pred = combiner(Xb)
            loss = ce(pred, yb)
            running_loss += loss.item()
            running_correct += pred.argmax(1).eq(yb).sum().item()
            n += len(yb)

        acc = running_correct / n
        accs.append(acc)
        # logger.print_and_log('Descriptor: %s, Acc: %s ' % (descriptor, acc), just_log=True)
        descriptors_accs['hnet_prediction_on_%s_%s' % (descriptor, pretext)] = acc

    logger.print_and_log(pretext + ' Average acc: %s' % (np.mean(accs)))

    metrics['hnet_acc_%s' % pretext] = np.mean(accs)
    metrics['hnet_median_%s' % pretext] = np.median(accs)
    metrics['hnet_std_%s' % pretext] = np.std(accs)
    metrics['hnet_SEM_%s' % pretext] = sem(accs)

    return metrics, descriptors_accs


def get_seperated_attributes(c1, c2, class2attribute):
    sa = [x + y == 1 for x, y in zip(class2attribute[c1], class2attribute[c2])]
    sa = set([x for x, y in enumerate(sa) if y])

    c1_att = set([x for x, y in enumerate(class2attribute[c1]) if y == 1])
    c2_att = set([x for x, y in enumerate(class2attribute[c2]) if y == 1])

    c1_att = list(c1_att - sa)
    c2_att = list(c2_att - sa)

    random.shuffle(c1_att)
    random.shuffle(c2_att)
    
    return c1_att, c2_att


def describe_label_with_atrributes(descriptor, labels2int, int2attribute, int2attributes_names, natt=4):
    domain, label1, label2 = descriptor.split(' ')
    l1_att, l2_att = get_seperated_attributes(labels2int[label1], labels2int[label2], int2attribute)
    l1_att, l2_att = [int2attributes_names[x] for x in l1_att] , [int2attributes_names[x] for x in l2_att]
    new_descriptor = 'The first class has: %s.[SEP]The second class has: %s.' % (' '.join(l1_att[:natt]), ' '.join(l1_att[:natt]))
    return new_descriptor


def train_resnet(train_loaders, val_loaders, labels2int):
    model = resnet18(pretrained=True).to('cuda')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    ce = torch.nn.CrossEntropyLoss()

    for i in (range(20)):
        model.train()
        accs = []
        for descriptor, loader in tqdm(train_loaders.items()):
            _, d1, d2 = descriptor.split(' ')
            swap = {0: labels2int[d1], 1: labels2int[d2]}
            running_correct, running_loss, n = 0.0, 0.0, 0.0
            for X, y, _ in loader:
                X, y = X.cuda(), torch.tensor([swap[x.item()] for x in y]).cuda()
                pred = model(X)
                loss = ce(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_correct += pred.argmax(1).eq(y).sum().item()
                n += len(y)
            accs.append(running_correct / n)
        print("Epoch %d" % i)
        eval_resnet(model, train_loaders, labels2int, prefix='train')
        eval_resnet(model, val_loaders, labels2int, prefix='test')

        torch.save(model.state_dict(), 'pretrain_resnet18_%s.pt' % np.mean(accs))
    return model


def get_optimizers(meta_model, args):
    optimizers = {'hnet_optimizer': torch.optim.SGD([p for (n, p) in meta_model.named_parameters() if 'hnet' in n],
                                     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay),
                  'feature_extractor_optimizer': torch.optim.SGD([p for (n, p) in meta_model.named_parameters() if 'feature_extractor' in n], 
                                                               lr=args.resnet_lr, momentum=args.resnet_momentum, weight_decay=args.resnet_weight_decay),
                  'text_encoder_optimizer': torch.optim.SGD([p for (n, p) in meta_model.named_parameters() if 'text_encoder' in n],
                                                            lr=args.text_encoder_lr, momentum=args.text_encoder_momentum, weight_decay=args.text_encoder_weight_decay)
                  }
    return optimizers


def get_text_encoder(encoder_type, encode_text_as, train_labels=None, p_negative=0.0, nattributes=4, dataset='AwA'):

    if (p_negative > 0.0 and encoder_type == 'SBERT') or (encoder_type =='TFIDF'):
        print('Using negative descriptions! p=%0.2f' % p_negative)
        if encoder_type == 'SBERT':
            text_encoder = myNSBERT(p_negative, nattributes)
            text_encoder_out_dim = 384
        if encoder_type == 'TFIDF':
            text_encoder = myTFIDF(dataset, p_negative, nattributes)
            text_encoder_out_dim = text_encoder.out_dim
        return text_encoder.cuda(), text_encoder_out_dim

    else:
        if encode_text_as == 'similarity':
            text_encoder = Similarity(train_labels, encoder_type, dataset=dataset)
            text_encoder_out_dim = len(train_labels)
        elif encode_text_as == 'dense':
            text_encoder = TextEncoder(encoder_type, dataset=dataset)
            text_encoder_out_dim = text_encoder.out_dim

        else:
            print('Error: Encoder type did not found!!!')
            text_encoder, text_encoder_out_dim = None, None

        return text_encoder.cuda(), text_encoder_out_dim


def get_data(dataset='AwA', tuned_resnet=0.0):
    data = {'train_visual_feature_extractor': dict(),
            'eval_visual_feature_extractor': dict(),
            'train_hn': dict(),
            'eval_hn': dict(),
            'zs_classes': dict()
            }

    if dataset == 'AwA':
        dataset_folder = 'awa'
        visual_features_dim = 512
    elif dataset == 'CUB':
        dataset_folder = 'cub'
        visual_features_dim = 512
    elif dataset == 'SUN':
        dataset_folder = 'sun'
        visual_features_dim = 512
    elif dataset == 'PC':
        dataset_folder = 'modelnet40'
        visual_features_dim = 256
    else:
        print('Error!! Do not find dataset!')
    resnet_features_type = '' if tuned_resnet == 0.0 else '_%s' % tuned_resnet
    for k in data.keys():
        (data[k]['X'], data[k]['class2idx']) = load_object('%s/features%s/%s.pkl' % (dataset_folder, resnet_features_type, k))
        data[k]['labels'] = list(data[k]['class2idx'].keys())
        data[k]['descriptions'] = \
            ['%s %s' % (l1, l2) for (l1, l2) in itertools.combinations(data[k]['labels'], 2)] + \
            ['%s %s' % (l2, l1) for (l1, l2) in itertools.combinations(data[k]['labels'], 2)]
    return data, visual_features_dim


def get_hnet(text_encoder_out_dim, resnet_out_dim, hn_hidden_dim, out_dim, hn_type):
    if hn_type == 'EV':
        return EVHN(text_encoder_out_dim, resnet_out_dim, hn_hidden_dim=hn_hidden_dim).cuda()
    elif hn_type == 'IVEV':
        return IVEVHN(text_encoder_out_dim, resnet_out_dim, hn_hidden_dim=hn_hidden_dim).cuda()
    elif hn_type == 'W':
        hnet = HNForResnet(out_dim * text_encoder_out_dim, resnet_out_dim, hn_hidden_dim=hn_hidden_dim, out_dim=out_dim).cuda()
        return hnet
    elif hn_type == 'WW':
        return WWHN(out_dim * text_encoder_out_dim, resnet_out_dim, hn_hidden_dim=hn_hidden_dim,
                    target_hidden_dim=hn_hidden_dim, target_out_dim=out_dim).cuda()
    else:
        print('Error! hn_type was not found!')
        exit()


def get_combiner(input_dim, hidden_dim, out_dim, hn_type):
    if hn_type in ('W','EV'):
        return ZSCombiner(input_dim=input_dim, out_dim=out_dim).cuda()
    elif hn_type in ('WW', 'IVEV'):
        return ZSCombiner_2l(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim).cuda()
    else:
        print('Error! hn_type combiner was not found!')
        exit()


if __name__ == '__main__':
    args = get_args()
    logger = Logger(filename='%s_%s_%s_%s' %(args.dataset, args.text_encoder, args.encode_text_as, args.log_file))
    logger.log_object(args, 'args')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, visual_features_dim = get_data(args.dataset, args.tuned_resnet)
    text_encoder, text_encoder_out_dim = get_text_encoder(args.text_encoder, args.encode_text_as,
                                                          data['train_visual_feature_extractor']['labels'],
                                                          args.p_negative, args.nattributes, args.dataset)

    hnet = get_hnet(text_encoder_out_dim, visual_features_dim, hn_hidden_dim=args.hnet_hidden_size, out_dim=args.out_dim,
                    hn_type=args.hn_type)
    combiner = get_combiner(input_dim=visual_features_dim, hidden_dim=args.hnet_hidden_size, out_dim=2, hn_type=args.hn_type)

    metrics = dict()
    hnet_optimizer = torch.optim.SGD(hnet.parameters(), lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay)
    ce = torch.nn.CrossEntropyLoss()

    metrics['best_val_acc'], metrics['best_test_acc'], metrics['best_test_acc_std'], metrics['best_test_acc_SEM'] = 0.0, 0.0, 0.0, 0.0
    for epoch in tqdm(range(args.hn_train_epochs)):
        hnet.train()

        all_descriptors = data['train_visual_feature_extractor']['descriptions'] + \
                          data['train_hn']['descriptions']
        random.shuffle(all_descriptors)
        for descriptor in all_descriptors[:100]:
            # loader = descriptor2loader[descriptor]
            # get data
            l0, l1 = descriptor.split(' ')
            if l0 in data['train_visual_feature_extractor']['class2idx']:
                c_class2idx = data['train_visual_feature_extractor']['class2idx']
                X0 = data['train_visual_feature_extractor']['X'][c_class2idx[l0]]
                X1 = data['train_visual_feature_extractor']['X'][c_class2idx[l1]]
            else:
                c_class2idx = data['train_hn']['class2idx']
                X0 = data['train_hn']['X'][c_class2idx[l0]]
                X1 = data['train_hn']['X'][c_class2idx[l1]]

            X = torch.cat([X0, X1])
            y = torch.tensor([0] * X0.shape[0] + [1] * X1.shape[0])
            perm_ind = torch.randperm(y.shape[0])

            descriptor_embed = text_encoder.forward(descriptor)
            descriptor_embed = [z.detach().float() for z in descriptor_embed]
            weights = hnet(descriptor_embed)

            combiner.load_state_dict(weights)
            inner_optim = torch.optim.SGD(combiner.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

            for i in range(args.inner_train_epochs):
                running_correct, running_loss, n = 0., 0., 0.

                combiner.train()

                j = 0
                batch_features = []
                while j * args.batch_size < perm_ind.shape[0]:
                    batch_ind = perm_ind[j * args.batch_size:(j + 1) * args.batch_size]
                    Xb, yb = X[batch_ind], y[batch_ind]
                    Xb, yb = Xb.to('cuda'), yb.to('cuda')
                    j += 1

                    pred = combiner(Xb)
                    loss = ce(pred, yb)
                    inner_optim.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(combiner.parameters(), 50)
                    inner_optim.step()

                    running_loss += loss.item()
                    running_correct += pred.argmax(1).eq(yb).sum().item()
                    n += len(yb)
                # print('%s Iteration: %d, Train acc: %s , Train loss: %s' % (descriptor, i, running_correct / n, running_loss / n))

            hnet_optimizer.zero_grad()
            grad_params = [p for p in hnet.parameters() if p.requires_grad]

            final_state = combiner.state_dict()
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

            grads = torch.autograd.grad(
                list(weights.values()), grad_params, grad_outputs=list(delta_theta.values()), allow_unused=True)

            # update hnet weights
            for p, g in zip(grad_params, grads):
                p.grad = g
            # torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
            hnet_optimizer.step()

        descriptors_accs = dict()
        for split in ['eval_visual_feature_extractor', 'eval_hn', 'zs_classes']:
            metrics , descriptors_accs[split] = eval_hresnet(hnet, text_encoder, combiner, data, logger, pretext=split, metrics=metrics)
        sys.stdout.flush()
        if metrics['hnet_acc_eval_hn'] > metrics['best_val_acc']:
            metrics['best_val_acc'] = metrics['hnet_acc_eval_hn']
            metrics['best_test_acc'] = metrics['hnet_acc_zs_classes']
            metrics['best_test_acc_std'] = metrics['hnet_std_zs_classes']
            metrics['best_test_acc_SEM'] = metrics['hnet_SEM_zs_classes']
            logger.log_model(hnet, add_to_name='best_eval_hn')
            try:
                logger.log_model(text_encoder, add_to_name='best_eval_hn_text_encoder')
            except:
                logger.print_and_log('Warning! Text encoder did not logged!')
    logger.log_object(descriptors_accs, 'descriptors_accs')

# AWA         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
