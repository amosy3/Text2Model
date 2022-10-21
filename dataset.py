import numpy as np
import random
from collections import defaultdict
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os
import utils
import torch
import pandas as pd
import itertools
from PIL import Image
from random import shuffle
from glob import glob


class ImageFolderFilterLabelsWithPaths(ImageFolder):
	def __init__(self, root, labels, transform=None):
		super().__init__(root)
		self.imgs = list(filter(lambda x: any(l in x[0] for l in labels), self.imgs))
		self.transform = transform

	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, index):
		path, y = self.imgs[index]
		X = self.loader(path)
		if self.transform is not None:
			X = self.transform(X)
		if self.target_transform is not None:
			y = self.target_transform(y)
		return X, y, path


class TripletLoaderFromPaths(Dataset):
	def __init__(self, path1, path2, path3, transform=None):
		self.path1 = path1
		self.path2 = path2
		self.path3 = path3
		self.transform = transform

	def __len__(self):
		return len(self.path1) + len(self.path2) + len(self.path3)

	def __getitem__(self, index):
		n1 = len(self.path1)
		n2 = len(self.path2)
		if index < n1:
			path = self.path1[index]
			y = 0
		elif (index >= n1) and (index < (n1+n2)):
			path = self.path2[index - n1]
			y = 1
		else:
			path = self.path3[index-n1-n2]
			y = 2

		with open(path, 'rb') as f:
			img = Image.open(f)
			X = img.convert('RGB')
			if self.transform is not None:
				X = self.transform(X)

		return X, y, path


class BinaryLoaderFromPaths(Dataset):
	def __init__(self, path1, path2, transform=None):
		self.path1 = path1
		self.path2 = path2
		self.transform = transform

	def __len__(self):
		return len(self.path1) + len(self.path2)

	def __getitem__(self, index):
		if index < len(self.path1):
			path = self.path1[index]
			y = 0
		else:
			path = self.path2[index-len(self.path1)]
			y = 1

		with open(path, 'rb') as f:
			img = Image.open(f)
			X = img.convert('RGB')
			if self.transform is not None:
				X = self.transform(X)

		return X, y, path



class ImageFolderWithPaths(ImageFolder):
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        X,y = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        return X, y, path


class PathDataset(Dataset):
	def __init__(self, paths):
		super().__init__(paths)
		self.paths = paths

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx):
		img_path = self.paths[idx]
		image = read_image(img_path) #Todo: find if need to convert to int
		label = img_path.split('/')[-2]
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label



def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
	src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
	src = torch.tensor(src, dtype=dtype).squeeze()
	return src


class ModelNetLoader(Dataset):
	def __init__(self, paths, labels):
		self.paths = paths
		self.labels = labels

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		path = self.paths[index]
		y = self.labels[index]

		with open(path, 'r') as f:
			src = f.read().split('\n')[:-1]
		if src[0] == 'OFF':
			src = src[1:]
		else:
			src[0] = src[0][3:]
		num_nodes, num_faces = [int(item) for item in src[0].split()[:2]]
		pc = parse_txt_array(src[1:1 + num_nodes]).numpy()
		choice = np.random.choice(pc.shape[0], 2500, replace=True)
		pc = pc[choice, :]
		pc = pc - np.expand_dims(np.mean(pc, axis=0), 0)  # center
		dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)), 0)
		pc = pc / dist  # scale
		pc = torch.tensor(pc)
		pc = pc.permute(1, 0)
		return pc, y, path


class BinaryModelNetLoader(Dataset):
	def __init__(self, paths1, paths2):
		self.paths1 = paths1
		self.paths2 = paths2

	def __len__(self):
		return len(self.paths1) + len(self.paths2)

	def __getitem__(self, index):

		if index < len(self.paths1):
			path = self.paths1[index]
			y = 0
		else:
			path = self.paths2[index - len(self.paths1)]
			y = 1


		with open(path, 'r') as f:
			src = f.read().split('\n')[:-1]
		if src[0] == 'OFF':
			src = src[1:]
		else:
			src[0] = src[0][3:]
		num_nodes, num_faces = [int(item) for item in src[0].split()[:2]]
		pc = parse_txt_array(src[1:1 + num_nodes]).numpy()
		choice = np.random.choice(pc.shape[0], 2500, replace=True)
		pc = pc[choice, :]
		pc = pc - np.expand_dims(np.mean(pc, axis=0), 0)  # center
		dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)), 0)
		pc = pc / dist  # scale
		pc = torch.tensor(pc)
		pc = pc.permute(1, 0)
		return pc, y, path


def get_paths_and_labels(classes, _class2int, split='train'):
	paths = []
	labels = []

	for _class in classes:
		class_paths = glob('/cortex/data/point_clouds/ModelNet_40_npy/%s/%s/*' % (_class, split))
		paths += class_paths
		labels += [_class2int[_class]] * len(class_paths)

	return paths, labels

#pass
def create_pool_loader(path, domains, labels, resize, batch_size, num_workers):
	transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor()])

	all_datasets = [ImageFolderFilterLabelsWithPaths('%s%s' % (path, domain), labels, transform=transform)
					for domain in domains]
	pool_dataset = torch.utils.data.ConcatDataset(all_datasets)
	pool_loader = DataLoader(pool_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
	return pool_loader



def create_loader_from_paths(paths):
	dataset = PathDataset(paths)
	return DataLoader(dataset, num_workers=6, batch_size=5, shuffle=True)



#pass
def get_gt_loaders(labels_split, descriptors, resize, batch_size, num_workers=0, inner_val_frac=0.1,
				   data_root='/cortex/data/images/DomainNet/domainnet', dataset='DomainNet'):
	transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor()])

	descriptor2loader = dict()
	for split in ['train', 'val', 'test']:
		label2paths = dict()
		for label in labels_split[split]:
			label2paths[label] = dict()
			paths = glob('%s/%s/*' % (data_root, label))
			if split == 'test':
				label2paths[label]['inner_val'] = paths
			else:
				n = int(inner_val_frac * len(paths))
				label2paths[label]['inner_train'] = paths[n:]
				label2paths[label]['inner_val'] = paths[:n]

		for descriptor in descriptors[split]:
			descriptor2loader[descriptor] = dict()
			domain, label1, label2 = descriptor.split(' ')

			# if dataset=='DomainNet':
			# 	paths1 = glob('%s/%s/%s/*' % (data_root, domain, label1))
			# 	paths2 = glob('%s/%s/%s/*' % (data_root, domain, label2))
			# if dataset=='AwA':
			# 	paths1 = glob('%s/%s/*' % (data_root, label1))
			# 	paths2 = glob('%s/%s/*' % (data_root, label2))

			inner_splits = ['inner_val'] if split == 'test' else ['inner_train', 'inner_val']
			for inner_split in inner_splits:
				binary_dataset = BinaryLoaderFromPaths(label2paths[label1][inner_split], label2paths[label2][inner_split], transform=transform)
				descriptor2loader[descriptor][inner_split] = DataLoader(binary_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, pin_memory=True)
	return descriptor2loader




def load_labels_name(path, use_shuffle=True, dataset='DomainNet'):
	txt_file = 'category.txt' if dataset == 'DomainNet' else '../classes.txt'
	file_name = '%s%s' % (path, txt_file)
	with open(file_name) as file:
		if dataset=='DomainNet':
			labels = [line.rstrip() for line in file.readlines()]
			int2label = {i: v for (i, v) in enumerate(labels)}
		else:
			classes = dict([line.rstrip().split(' ')[-1].split('\t') for line in file.readlines()])
			int2label = {int(k) - 1: v for k, v in classes.items()}
			labels = list(int2label.values())
	if use_shuffle:
		shuffle(labels)

	labels_split = dict()
	if dataset == 'DomainNet':
		pass
	else:
		with open('%s../trainclasses.txt' % path) as file:
			labels_split['train'] = [line.rstrip() for line in file.readlines()]
		with open('%s../testclasses.txt' % path) as file:
			labels_split['test'] = [line.rstrip() for line in file.readlines()]

	return labels, int2label, labels_split


def generate_descriptions_from_labels(labels_split, domains, method, corruptions=None):
	all_queries = dict()
	if method == 'random':
		for i in range(nsamples):
			domain = random.choice(domains)
			label1 = random.choice(labels)
			label2 = random.choice(labels)

			if corruptions is None:
				all_queries.append('%s %s %s' % (domain, label1, label2))
			else:
				corruption = random.choice(corruptions)
				all_queries.append('%s %s %s %s' % (domain, label1, label2, corruption))

	elif method == 'all_pairs':
		for split in ['train', 'val', 'test']:
			s1 = ['%s %s %s' % (domain, l1, l2) for (l1, l2) in
								  list(itertools.combinations(labels_split[split], 2)) for domain in domains]
			s2 = ['%s %s %s' % (domain, l2, l1) for (l1, l2) in
								  list(itertools.combinations(labels_split[split], 2)) for domain in domains]
			all_queries[split] = s1 + s2

	elif method == 'single':
		for i in range(nsamples):
			domain = random.choice(domains)
			label = random.choice(labels)
			all_queries.append('%s %s' % (domain, label))

		else:
			assert False, "Description generation method is not exist!"
	return all_queries


def split_descriptors(labels, descriptors, domains, nzs, test_frac):
	shuffle(descriptors)

	# save labels to evaluate zero shot prediction
	labels_for_zero_shot = labels[:nzs]
	labels_for_training = labels[nzs:]

	#Remove descriptors that contain labels that use for ZS setup
	train_descriptors = list(filter(lambda q: not (any([x in q for x in labels_for_zero_shot])), descriptors))

	zs1_keys = ["%s %s %s" % (d, l1, l2) for l1 in labels_for_zero_shot for l2 in
				labels_for_training for d in domains]
	all_zs2_combinations = list(itertools.combinations(labels_for_zero_shot, 2))
	zs2_keys = ['%s %s %s' % (d, l1, l2) for (l1, l2) in all_zs2_combinations for d in domains]

	# train/test split
	ntest = int(len(train_descriptors) * test_frac)
	train_keys, test_keys = train_descriptors[ntest:], train_descriptors[:ntest]

	return train_keys, test_keys, zs1_keys, zs2_keys


def arange_splits(labels_split, val_frac):
	n = int(val_frac * len(labels_split['train']))
	labels_split['train'], labels_split['val'] = labels_split['train'][n:], labels_split['train'][:n]
	for split in ['train', 'val', 'test']:
		print('Split %s use labels: %s' % (split, labels_split[split]))
	return labels_split


def get_hn_loaders(args):

	if args.dataset == 'AO_clevr':
		resnet_train = pd.read_pickle("hard_ao_clevr_pkls/resnet_train.pkl")
		resnet_val = pd.read_pickle("hard_ao_clevr_pkls/resnet_val.pkl")
		hres_train = pd.read_pickle("hard_ao_clevr_pkls/hres_train.pkl")
		hres_val = pd.read_pickle("hard_ao_clevr_pkls/hres_val.pkl")
		test_hresnet = pd.read_pickle("hard_ao_clevr_pkls/test_hresnet.pkl")

		labels_split = dict()
		labels_split['train'] = list(resnet_train['image_description'].unique())
		labels_split['val'] = list(hres_train['image_description'].unique())
		labels_split['test'] = list(test_hresnet['image_description'].unique())

		for split in ['train', 'val', 'test']:
			print('Split %s use labels: %s' % (split, labels_split[split]))

		descriptions = dict()
		for split in ['train', 'val', 'test']:
			descriptions[split] = ['%s %s' % (l1, l2) for (l1, l2) in itertools.combinations(labels_split[split], 2)]

		resize = 96
		transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor()])
		all_loaders = dict()

		key2df = {'train_resnet': (resnet_train, 'train'),
				  'eval_resnet': (resnet_val, 'train'),
				  'train_hres': (hres_train, 'val'),
				  'eval_hres': (hres_val, 'val'),
				  'test_hres': (test_hresnet, 'test'),
				  }

		for k, (df, s) in key2df.items():
			descriptor2loader = dict()
			for descriptor in descriptions[s]:
				d1, d2, d3, d4 = descriptor.split(' ')
				l1 = '%s %s' % (d1, d2)
				l2 = '%s %s' % (d3, d4)

				mask1 = (df['image_description'] == l1)
				mask2 = (df['image_description'] == l2)
				data_folder = '/cortex/data/images/' #'/mnt/dsi_vol1/users/amosy/' #
				paths1 = [data_folder + 'AO_clevr/ao_clevr/images/' + p for p in list(df['image_filename'][mask1].values)]
				paths2 = [data_folder + 'AO_clevr/ao_clevr/images/' + p for p in list(df['image_filename'][mask2].values)]

				binary_dataset = BinaryLoaderFromPaths(paths1, paths2, transform=transform)
				descriptor2loader[descriptor] = DataLoader(binary_dataset, num_workers=20, batch_size=32,
														   shuffle=True, pin_memory=True)
			all_loaders[k] = descriptor2loader

		return all_loaders, None, labels_split #descriptions

	elif args.dataset == 'CUB':
		root_path = '/cortex/data/images/CUB/CUB_CVPR2016'

		with open('%s/allclasses.txt' % root_path) as file:
			all_classes_raw = [line.rstrip() for line in file.readlines()]

		with open('%s/testclasses.txt' % root_path) as file:
			test_classes = [line.rstrip() for line in file.readlines()]

		train_classes_raw = [x for x in all_classes_raw if x not in test_classes]

		labels_split = dict()
		labels_split['train'] = train_classes_raw[:100]
		labels_split['val'] = train_classes_raw[100:]
		labels_split['test'] = test_classes

		labels_split['train'][15] = '022.Chuck_will_Widow'
		labels_split['train'][41] = '061.Heermann_Gull'
		labels_split['train'][46] = '067.Anna_Hummingbird'
		labels_split['train'][66] = '093.Clark_Nutcracker'
		labels_split['train'][79] = '113.Baird_Sparrow'
		labels_split['train'][80] = '115.Brewer_Sparrow'
		labels_split['train'][85] = '122.Harris_Sparrow'
		labels_split['train'][86] = '123.Henslow_Sparrow'
		labels_split['train'][87] = '124.Le_Conte_Sparrow'
		labels_split['train'][88] = '125.Lincoln_Sparrow'
		labels_split['val'][33] = '178.Swainson_Warbler'
		labels_split['test'][4] = '009.Brewer_Blackbird'
		labels_split['test'][6] = '023.Brandt_Cormorant'
		labels_split['test'][28] = '098.Scott_Oriole'

		all_loaders = {'train_resnet': dict(),
					   'eval_resnet': dict(),
					   'train_hres': dict(),
					   'eval_hres': dict(),
					   'test_hres': dict(),
					   }

		transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
		for split in ['train', 'val', 'test']:
			for (d1, d2) in list(itertools.combinations(labels_split[split], 2)):
				d1_paths = glob('%s/../CUB_200_2011/images/%s*/*.jpg' % (root_path, d1.split('.')[0]))
				d2_paths = glob('%s/../CUB_200_2011/images/%s*/*.jpg' % (root_path, d2.split('.')[0]))
				n1, n2 = int(0.1 * len(d1_paths)), int(0.1 * len(d2_paths))

				if split in ('train', 'val'):
					binary_dataset = BinaryLoaderFromPaths(d1_paths[n1:], d2_paths[n2:], transform)
					eval_binary_dataset = BinaryLoaderFromPaths(d1_paths[:n1], d2_paths[:n2], transform)
					if split == 'train':
						k_train, k_eval = 'train_resnet', 'eval_resnet'
					if split == 'val':
						k_train, k_eval = 'train_hres', 'eval_hres'

					all_loaders[k_train]['%s %s' % (d1, d2)] = DataLoader(binary_dataset, num_workers=20, batch_size=32,
																		  shuffle=True, pin_memory=True)
					all_loaders[k_eval]['%s %s' % (d1, d2)] = DataLoader(eval_binary_dataset, num_workers=20,
																		 batch_size=32, shuffle=True, pin_memory=True)

				else:
					binary_dataset = BinaryLoaderFromPaths(d1_paths, d2_paths, transform)
					all_loaders['test_hres']['%s %s' % (d1, d2)] = DataLoader(binary_dataset, num_workers=20,
																			  batch_size=32, shuffle=True,
																			  pin_memory=True)

		return all_loaders, None, labels_split

	elif args.dataset == 'PC':
		all_classes = ['airplane', 'bowl', 'desk', 'keyboard', 'person', 'sofa', 'tv_stand', 'bathtub', 'car', 'door',
					   'lamp', 'piano', 'stairs', 'vase', 'bed', 'chair', 'dresser', 'laptop', 'plant', 'stool',
					   'wardrobe',
					   'bench', 'cone', 'flower_pot', 'mantel', 'radio', 'table', 'xbox', 'bookshelf', 'cup',
					   'glass_box',
					   'monitor', 'range_hood', 'tent', 'bottle', 'curtain', 'guitar', 'night_stand', 'sink', 'toilet']

		class2int = {v: k for k, v in enumerate(all_classes)}
		labels_split = dict()
		labels_split['train'] = all_classes[:26]
		labels_split['val'] = all_classes[26:33]
		labels_split['test'] = all_classes[33:]
		for split in ['train', 'val', 'test']:
			print('Split %s use labels: %s' % (split, labels_split[split]))

		split2descriptors = \
			{'train_resnet': (['%s %s' % (d1, d2) for (d1, d2) in list(itertools.combinations(labels_split['train'], 2))], 'train'),
			  'eval_resnet': (['%s %s' % (d1, d2) for (d1, d2) in list(itertools.combinations(labels_split['train'], 2))], 'test'),
			  'train_hres': (['%s %s' % (d1, d2) for (d1, d2) in list(itertools.combinations(labels_split['val'], 2))], 'train'),
			  'eval_hres': (['%s %s' % (d1, d2) for (d1, d2) in list(itertools.combinations(labels_split['val'], 2))], 'test'),
			  'test_hres': (['%s %s' % (d1, d2) for (d1, d2) in list(itertools.combinations(labels_split['test'], 2))], 'train and test'),
		 	 }

		# for each split:
			#get all descriptors
			#get all paths
			#create a DL

		all_loaders = dict()

		for k, (descriptors, split) in split2descriptors.items():
			descriptor2loader = dict()
			for descriptor in descriptors:
				d1, d2 = descriptor.split(' ')
				if split == 'train and test':
					d1_paths = glob('/cortex/data/point_clouds/ModelNet40/%s/*/*' % (d1))
					d2_paths = glob('/cortex/data/point_clouds/ModelNet40/%s/*/*' % (d2))
				else:
					d1_paths = glob('/cortex/data/point_clouds/ModelNet40/%s/%s/*' % (d1, split))
					d2_paths = glob('/cortex/data/point_clouds/ModelNet40/%s/%s/*' % (d2, split))

				binary_dataset = BinaryModelNetLoader(d1_paths, d2_paths)
				descriptor2loader[descriptor] = DataLoader(binary_dataset, num_workers=20, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)
			all_loaders[k] = descriptor2loader

		return all_loaders, None, labels_split #descriptions

	else:
		labels, int2label, labels_split = load_labels_name(args.path, use_shuffle=True, dataset=args.dataset)
		labels_split = arange_splits(labels_split, args.val_frac)

		descriptors = generate_descriptions_from_labels(labels_split, args.domains, args.descriptor_generator)

		gt_descriptor2loader = get_gt_loaders(labels_split, descriptors, args.resize, args.batch_size, inner_val_frac=0.1,
											  num_workers=args.num_workers,
											  data_root=args.path, dataset=args.dataset) #'/cortex/data/images/DomainNet/domainnet'
		# train_des, test_des, z1_des, z2_des = split_descriptors(labels, descriptors, args.domains, args.nzs, args.test_frac)

		all_loaders = {'train_resnet': {k: gt_descriptor2loader[k]['inner_train'] for k in descriptors['train']},
					   'eval_resnet': {k: gt_descriptor2loader[k]['inner_val'] for k in descriptors['train']},
					   'train_hres': {k: gt_descriptor2loader[k]['inner_train'] for k in descriptors['val']},
					   'eval_hres': {k: gt_descriptor2loader[k]['inner_val'] for k in descriptors['val']},
					   'test_hres': {k: gt_descriptor2loader[k]['inner_val'] for k in descriptors['test']}
					   }

		class2attribute = None
		int2attributes_names = None
		if args.dataset == 'AwA':
			with open('%s../predicate-matrix-binary.txt' % args.path) as file:
				class2attribute_raw = [line.rstrip().split(' ') for line in file.readlines()]
			class2attribute = dict()
			for i in range(50):
				class2attribute[i] = list(map(int, class2attribute_raw[i]))

			with open('%s../predicates.txt' % args.path) as file:
				int2attributes_names = dict([line.rstrip().split(' ')[-1].split('\t') for line in file.readlines()])
			int2attributes_names = {int(k) - 1: v for k, v in int2attributes_names.items()}

		labels2int = {v: k for k, v in int2label.items()}

		data_maps = {'labels2int': labels2int,
					 'int2label': int2label,
					 'class2attribute': class2attribute,
					 'int2attributes_names': int2attributes_names
					 }

		return all_loaders, data_maps, labels_split


def get_triplets_loaders(args):
	labels, int2label, labels_split = load_labels_name(args.path, use_shuffle=True, dataset=args.dataset)
	labels_split = arange_splits(labels_split, args.val_frac)

	descriptors = dict()
	for split in ['train', 'val', 'test']:
		descriptors[split] = ['%s %s %s' % (l1, l2, l3) for (l1, l2, l3) in list(itertools.combinations(labels_split[split], 3))]

	transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
	inner_val_frac = 0.1
	descriptor2loader = dict()
	for split in ['train', 'val', 'test']:
		label2paths = dict()
		for label in labels_split[split]:
			label2paths[label] = dict()
			paths = glob('%s/%s/*' % (args.path, label))
			if split == 'test':
				label2paths[label]['inner_val'] = paths
			else:
				n = int(inner_val_frac * len(paths))
				label2paths[label]['inner_train'] = paths[n:]
				label2paths[label]['inner_val'] = paths[:n]

		for descriptor in descriptors[split]:
			descriptor2loader[descriptor] = dict()
			label1, label2, label3 = descriptor.split(' ')

			inner_splits = ['inner_val'] if split == 'test' else ['inner_train', 'inner_val']
			for inner_split in inner_splits:
				triplet_dataset = TripletLoaderFromPaths(label2paths[label1][inner_split],
														label2paths[label2][inner_split],
														label2paths[label3][inner_split],
														transform=transform)
				descriptor2loader[descriptor][inner_split] = DataLoader(triplet_dataset, num_workers=args.num_workers,
																		batch_size=args.batch_size, shuffle=True,
																		pin_memory=True)



	all_loaders = {'train_resnet': {k: descriptor2loader[k]['inner_train'] for k in descriptors['train']},
				   'eval_resnet': {k: descriptor2loader[k]['inner_val'] for k in descriptors['train']},
				   'train_hres': {k: descriptor2loader[k]['inner_train'] for k in descriptors['val']},
				   'eval_hres': {k: descriptor2loader[k]['inner_val'] for k in descriptors['val']},
				   'test_hres': {k: descriptor2loader[k]['inner_val'] for k in descriptors['test']}
				   }


	class2attribute = None
	int2attributes_names = None
	if args.dataset == 'AwA':
		with open('%s../predicate-matrix-binary.txt' % args.path) as file:
			class2attribute_raw = [line.rstrip().split(' ') for line in file.readlines()]
		class2attribute = dict()
		for i in range(50):
			class2attribute[i] = list(map(int, class2attribute_raw[i]))

		with open('%s../predicates.txt' % args.path) as file:
			int2attributes_names = dict([line.rstrip().split(' ')[-1].split('\t') for line in file.readlines()])
		int2attributes_names = {int(k) - 1: v for k, v in int2attributes_names.items()}

	labels2int = {v: k for k, v in int2label.items()}

	data_maps = {'labels2int': labels2int,
				 'int2label': int2label,
				 'class2attribute': class2attribute,
				 'int2attributes_names': int2attributes_names
				 }

	return all_loaders, data_maps, labels_split







