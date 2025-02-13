# This is the code to implement VLC on MobileNet-V2 model
import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim
import numpy as np
from tqdm import tqdm
import random
import wandb
import os
import re
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from PIL import ImageFile
import argparse



# os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda:0')  # Device configuration



def cal_Pbn_entropy(model,bn_list):
	# Function to get BN layer parameters 
	cdf_0 = {}	
	beta = {}
	for name, module in model.named_modules():
		if name in bn_list:
			Gamma = torch.abs(module.weight.data)
			Beta = module.bias.data
			normal_part = 0.5 * (1 + torch.erf((0 - Beta) / (Gamma * np.sqrt(2))))
			zero_gamma_condition = Gamma == 0
			cdf_0[name]= torch.where(zero_gamma_condition, (0 >= Beta).float(), normal_part)
			beta[name] = Beta
	return (cdf_0, beta)


def train(model, epoch, optimizer,train_loader):
	print('\nEpoch : %d' % epoch)
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	loss_fn = torch.nn.CrossEntropyLoss()
	for data in tqdm(train_loader):
		inputs, labels = data[0].to(device), data[1].to(device)
		outputs = model(inputs)
		loss = loss_fn(outputs, labels)      
		total_loss = loss    
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()		
		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()        
		running_loss += total_loss.item()	
	train_loss = running_loss / len(train_loader)
	accu = (100.0 * correct / total)
	print('Train Loss: %.3f | Accuracy: %.3f' % (train_loss, accu))	
	return accu, train_loss


def val(model,val_loader):
	model.eval()
	running_loss=0
	correct=0
	total=0    
	loss_fn=torch.nn.CrossEntropyLoss()
	with torch.no_grad():
		for data in tqdm(val_loader):
			images,labels=data[0].to(device),data[1].to(device)
			outputs=model(images)
			loss= loss_fn(outputs,labels)
			running_loss+=loss.item()     
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()  
	test_loss=running_loss/len(val_loader)
	accu=100.*correct/total	
	print('Val Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
	return(accu, test_loss)


def test(model,test_loader):
	model.eval()	
	running_loss=0
	correct=0
	total=0    
	loss_fn=torch.nn.CrossEntropyLoss()
	with torch.no_grad():
		for data in tqdm(test_loader):
			images,labels=data[0].to(device),data[1].to(device)
			outputs=model(images)
			loss= loss_fn(outputs,labels)
			running_loss+=loss.item()       
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()   
	test_loss=running_loss/len(test_loader)
	accu=100.*correct/total
	print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
	return(accu, test_loss)


def testVal(model,val_loader):
	model.eval()	
	running_loss=0
	correct=0
	total=0    
	loss_fn=torch.nn.CrossEntropyLoss()
	with torch.no_grad():
		for data in tqdm(val_loader):
			images,labels=data[0].to(device),data[1].to(device)
			outputs=model(images)
			loss= loss_fn(outputs,labels)
			running_loss+=loss.item()     
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()  
			break
	test_loss=running_loss
	accu=100.*correct/total	
	print('testVal_loss Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
	return(accu, test_loss)


def replace_relu6_inplace(model):
	for name, module in model.named_children():
		if type(module) == torch.nn.ReLU6:
			setattr(model, name, torch.nn.ReLU6(inplace=False))
		else:
			replace_relu6_inplace(module) 





def main():

	parser = argparse.ArgumentParser(description='TLC')      
	parser.add_argument('--dataset', default='CIFAR-10', help='dataset')
	parser.add_argument('--DATA_DIR', default='~/data/cifar-10/', help='data_root')
	parser.add_argument('--seed',  type=int, default=43, help='seed')

	args = parser.parse_args()

	# set seeds
	torch.manual_seed(args.seed)
	os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.use_deterministic_algorithms(True)

	# dataset
	if args.dataset == 'CIFAR-10':
		epochs = 160
		learning_rate = 0.1
		momentum = 0.9
		gamma=0.1
		weight_decay = 1e-4
		milestones=[80,120]
		batch_size = 128
		size_train_set = 0.9
		transform = transforms.Compose([transforms.RandomHorizontalFlip(),
												transforms.RandomCrop(32,4),
												transforms.ToTensor(),
												transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
		train_dataset = torchvision.datasets.CIFAR10(root=args.DATA_DIR,
													train=True,
													transform=transform,
													download=True)
		train_size = int(size_train_set * len(train_dataset))
		val_size = len(train_dataset) - train_size
		train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
		test_dataset = torchvision.datasets.CIFAR10(root=args.DATA_DIR,
													train=False,
													transform=transforms.Compose([transforms.ToTensor(),
																				transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
													download=True)
		# Data loader
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
												batch_size=batch_size, 
												shuffle=True,
												num_workers = 8)
		val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
												batch_size=batch_size, 
												shuffle=False,
												num_workers = 8)
		test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
												batch_size=batch_size, 
												shuffle=False,
												num_workers = 8)
		num_classes = 10

	elif args.dataset == 'Tiny-ImageNet-200':
		epochs = 160
		learning_rate = 0.1
		momentum = 0.9
		gamma=0.1
		weight_decay = 1e-4
		milestones=[80,120]
		batch_size = 128
		DATA_DIR = args.DATA_DIR # Original images come in shapes of [3,64,64]
		TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
		VALID_DIR = os.path.join(DATA_DIR, 'val')
		TEST_DIR = os.path.join(DATA_DIR, 'test')
		# Define transformation sequence for image pre-processing
		transform_train = transforms.Compose([
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		transform_val = transforms.Compose([
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		transform_test = transforms.Compose([
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
		train_loader = DataLoader(train_dataset,
								batch_size=batch_size, 
								shuffle=True, 
								num_workers=8)
		val_dataset = torchvision.datasets.ImageFolder(VALID_DIR, transform=transform_val)
		val_loader = DataLoader(val_dataset,
								batch_size=batch_size, 
								shuffle=False, 
								num_workers=8)
		test_dataset = torchvision.datasets.ImageFolder(TEST_DIR, transform=transform_test)
		test_loader = DataLoader(test_dataset,
								batch_size=batch_size, 
								shuffle=False, 
								num_workers=8)
		num_classes = 200

	elif args.dataset == 'ImageNet-1000':
		epochs = 90
		learning_rate = 0.1
		momentum = 0.9
		gamma=0.1
		weight_decay = 1e-4
		milestones=[30,60]
		batch_size = 128
		class ImageTransform(dict):
			def __init__(self):
				super().__init__(
					{"train": self.build_train_transform(), "val": self.build_val_transform()}
				)
			def build_train_transform(self):
				t = transforms.Compose(
					[
						transforms.RandomResizedCrop(256),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
					]
				)
				return t
			def build_val_transform(self):
				t = transforms.Compose(
					[
						transforms.Resize(256),
						transforms.CenterCrop(224),
						transforms.ToTensor(),
						transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
					]
				)
				return t
		dataset = {
					"train_val":  torchvision.datasets.ImageFolder(root=args.DATA_DIR+'/train/', transform=ImageTransform()["train"]),                           
					"test": torchvision.datasets.ImageFolder(root=args.DATA_DIR+'/val/',transform=ImageTransform()["val"]),
				}
		indices = np.arange(len(dataset["train_val"]))
		np.random.shuffle(indices)
		split_idx = int(0.95 * len(dataset["train_val"]))  
		train_indices, val_indices = indices[:split_idx], indices[split_idx:]
		train_dataset = Subset(dataset["train_val"], train_indices)
		val_dataset = Subset(dataset["train_val"], val_indices)
		train_loader = DataLoader(train_dataset,
								batch_size=batch_size, 
								shuffle=True, 
								num_workers=4)
		val_loader = DataLoader(val_dataset,
								batch_size=batch_size, 
								shuffle=False, 
								num_workers=4)
		test_loader = DataLoader(dataset["test"],
								batch_size=batch_size, 
								shuffle=False, 
								num_workers=4)
		num_classes=1000

	elif args.dataset == 'PACS':
		epochs = 30
		learning_rate = 0.001
		momentum = 0.9
		gamma=0.1
		weight_decay = 5e-4
		milestones=[24]
		batch_size =16		
		data_path = args.DATA_DIR
		ImageFile.LOAD_TRUNCATED_IMAGES = True
		transform = transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly cropping the images
			transforms.RandomHorizontalFlip(),  # Randomly apply horizontal flipping
			transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),  # Random color jittering
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		datasets = [ImageFolder(os.path.join(data_path, domain), transform=transform) 
						for domain in ['cartoon', 'art_painting', 'photo', 'sketch']]
		dataset = torch.utils.data.ConcatDataset(datasets)
		indices = np.arange(len(dataset))
		np.random.shuffle(indices)
		train_split = int(0.8 * len(dataset))  # 80% for training
		val_test_split = int(0.9 * len(dataset))  # 10% for validation, 10% for testing
		train_indices = indices[:train_split]
		val_indices = indices[train_split:val_test_split]
		test_indices = indices[val_test_split:]
		train_dataset = Subset(dataset, train_indices)
		val_dataset = Subset(dataset, val_indices)
		test_dataset = Subset(dataset, test_indices)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
		num_classes = len(dataset.datasets[0].classes)

	elif args.dataset == 'VLCS':
		epochs = 30
		learning_rate = 0.001
		momentum = 0.9
		gamma=0.1
		weight_decay = 5e-4
		milestones=[24]
		batch_size =16		
		data_path = args.DATA_DIR
		ImageFile.LOAD_TRUNCATED_IMAGES = True
		transform = transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly cropping the images
			transforms.RandomHorizontalFlip(),  # Randomly apply horizontal flipping
			transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),  # Random color jittering
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		datasets = [ImageFolder(os.path.join(data_path, domain), transform=transform) 
						for domain in ['VOC2007', 'SUN09', 'LabelMe', 'Caltech101']]
		dataset = torch.utils.data.ConcatDataset(datasets)
		indices = np.arange(len(dataset))
		np.random.shuffle(indices)
		train_split = int(0.8 * len(dataset))  # 80% for training
		val_test_split = int(0.9 * len(dataset))  # 10% for validation, 10% for testing
		train_indices = indices[:train_split]
		val_indices = indices[train_split:val_test_split]
		test_indices = indices[val_test_split:]
		train_dataset = Subset(dataset, train_indices)
		val_dataset = Subset(dataset, val_indices)
		test_dataset = Subset(dataset, test_indices)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
		num_classes = len(dataset.datasets[0].classes)	


	save_path = './' +'MobileNet/'+args.dataset+'/'
	if not os.path.exists(save_path+'/model_save/'):
		os.makedirs(save_path+'/model_save/')

	# model
	if args.dataset == 'CIFAR-10':
		from cifar10_models.mobilenetv2 import MobileNetV2
		model = MobileNetV2()
	else:
		model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
		model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
	replace_relu6_inplace(model)
	model.to(device)


	bn_list=[]
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.BatchNorm2d) and 'downsample' not in name:
			bn_list.append(name)
			

	val_acc, val_loss = val(model, val_loader)
	test_acc, test_loss = test(model,test_loader)

	data_columns = ['Lay.rem'] + ["train_acc"] + ["val_acc"] + ["test_acc"] 
	accumulated_data = pd.DataFrame(columns=data_columns)


	for it in range(0, 20):
		print('it', it)
		if it > 0:  # the first training is vanilla training
			cdf_0, beta = cal_Pbn_entropy(model, bn_list)

			temp_layer = []
			layers_to_replace = []
			layers_to_prune=[]
			for key in cdf_0.keys():
				temp_layer.append(key)

			beta_conv = {}
			previous_conv_name = None
			for name, module in model.named_modules():
				if type(module) == torch.nn.Conv2d or type(module) == torch.nn.Linear:
					previous_conv_name = name
					conv_layer = module
				elif name in temp_layer:
					beta_conv[previous_conv_name] = beta[name]
					bn_layer = module

			loss_wo_layer = {}
			acc_wo_layer = {}
			loss_gap={}
			acc_gap={}

			testVal_acc, testVal_loss = testVal(model,val_loader)
			loss_wo_layer['no']=testVal_loss
			acc_wo_layer['no'] = testVal_acc

			# rank layers by importance
			for name_conv, module in model.named_modules():
				if name_conv in beta_conv.keys() :
					print(name_conv)
					bn_close = []
					layers_to_replace = []
					model_copy = torch.load(model_path).to(device)
					previous_conv = False
					Previous_bn = False
					for name, module in model_copy.named_modules():
						if name == name_conv:
							previous_conv_name = name
							previous_conv = True
						elif name in bn_list and previous_conv == True:
							bn_close.append(name)
							previous_conv = False
							Previous_bn = True
						elif type(module) == torch.nn.ReLU6 and Previous_bn == True:
							layers_to_replace.append(name)
							Previous_bn = False
					
					for name, module in model_copy.named_modules():
						layer_mask = []
						if name == name_conv:
							for i in range(beta_conv[name_conv].size()[0]):
								if beta_conv[name_conv][i] > 0 :
									custom_mask = torch.ones(module.weight.data[i].size()).cpu().numpy()
									layer_mask.append(custom_mask) 
								else:
									custom_mask = torch.zeros(module.weight[i].size()).cpu().numpy()                                                       
									layer_mask.append(custom_mask) 
							layer_mask = torch.Tensor(layer_mask).to(device)
							torch.nn.utils.prune.custom_from_mask(module, name="weight", mask=layer_mask)

					for name, module in model_copy.named_modules():
						if name in bn_close:
							for i in range(beta[name].size()[0]):
								if beta[name][i] < 0 :
									module.weight.data[i] = 0                                                        
									module.bias.data[i] = 0		

					for name in layers_to_replace:  
						change_name= re.sub(r'\.(\d+)', r'[\1]', name)
						exec(f'model_copy.{change_name} = nn.Identity()')

					testVal_acc, testVal_loss = testVal(model_copy,val_loader)
					loss_wo_layer[name_conv]=testVal_loss
					acc_wo_layer[name_conv]=testVal_acc
					loss_gap[name_conv] = testVal_loss - loss_wo_layer['no']
					acc_gap[name_conv] =  acc_wo_layer['no'] - testVal_acc 
										
			sorted_loss_gap = dict(sorted(loss_gap.items(), key=lambda item: item[1]))

			# attempt remove multiple layers at once
			layers_to_prune = []
			if min(sorted_loss_gap.values()) < 0:
				for num_combine in range(2,len(loss_gap)):
					combine_remove_layer = [key for key, value in sorted(sorted_loss_gap.items(), key=lambda item: item[1])][:num_combine]
					print('remove layers',combine_remove_layer)
					model_com_rem = torch.load(model_path).to(device)
					bn_close = []
					layers_to_replace = []
					previous_conv = False
					Previous_bn = False					
					for name, module in model_com_rem.named_modules():
						if name in combine_remove_layer :
							previous_conv_name = name
							previous_conv = True
							layer_mask = []
							for i in range(beta_conv[name].size()[0]):
								if beta_conv[name][i] > 0 :
									custom_mask = torch.ones(module.weight.data[i].size()).cpu().numpy()
									layer_mask.append(custom_mask) 
								else:
									custom_mask = torch.zeros(module.weight[i].size()).cpu().numpy()                                                       
									layer_mask.append(custom_mask) 
							layer_mask = torch.Tensor(layer_mask).to(device)
							torch.nn.utils.prune.custom_from_mask(module, name="weight", mask=layer_mask)
						elif name in bn_list and previous_conv == True:
							bn_close.append(name)
							previous_conv = False
							Previous_bn = True
						elif type(module) == torch.nn.ReLU6 and Previous_bn == True:
							layers_to_replace.append(name)
							Previous_bn = False
					for name, module in model_com_rem.named_modules():
						if name in bn_close:
							for i in range(beta[name].size()[0]):
								if beta[name][i] < 0 :
									module.weight.data[i] = 0                                                          
									module.bias.data[i] = 0		


					for name in layers_to_replace:  
						change_name= re.sub(r'\.(\d+)', r'[\1]', name)
						exec(f'model_com_rem.{change_name} = nn.Identity()')	

					testVal_acc, testVal_loss = testVal(model_com_rem,val_loader)

					if testVal_loss - loss_wo_layer['no'] > 0:
						layers_to_prune = combine_remove_layer
						break

			if not layers_to_prune and loss_gap:
				min_key = min(loss_gap, key=loss_gap.get)
				layers_to_prune.append(min_key)

			# Permanently removing layers from a model
			model = torch.load(model_path).to(device)
			bn_close = []
			layers_to_replace = []
			previous_conv = False
			Previous_bn = False				
			for name, module in model.named_modules():
				if name in layers_to_prune :
					previous_conv_name = name
					previous_conv = True
					layer_mask = []
					for i in range(beta_conv[name].size()[0]):
						if beta_conv[name][i] > 0 :
							custom_mask = torch.ones(module.weight.data[i].size()).cpu().numpy()
							layer_mask.append(custom_mask) 
						else:
							custom_mask = torch.zeros(module.weight[i].size()).cpu().numpy()                                                    
							layer_mask.append(custom_mask) 
					layer_mask = torch.Tensor(layer_mask).to(device)
					torch.nn.utils.prune.custom_from_mask(module, name="weight", mask=layer_mask)
				elif name in bn_list and previous_conv == True:
					bn_close.append(name)
					previous_conv = False
					Previous_bn = True
				elif type(module) == torch.nn.ReLU6 and Previous_bn == True:
					layers_to_replace.append(name)
					Previous_bn = False

			for name, module in model.named_modules():
				if name in bn_close:
					bn_list.remove(name)
					for i in range(beta[name].size()[0]):
						if beta[name][i] < 0 :
							module.weight.data[i] = 0                                        
							module.bias.data[i] = 0		

			for name in layers_to_replace:  
				change_name= re.sub(r'\.(\d+)', r'[\1]', name)
				exec(f'model.{change_name} = nn.Identity()')

		# number of Identity layers
		Iden_num = 0
		Iden_list = []
		for name, module in model.named_modules():
			if type(module) == torch.nn.Identity:
				Iden_num +=1
				Iden_list.append(name)


		name_of_run = "it_"+str(it) + '_Iden_' + str(Iden_num)
		name_model = name_of_run

		final_val_acc = 0
		optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)	

		# training
		for epoch in range(1,epochs+1):
			train_acc, train_loss = train(model, epoch, optimizer,train_loader)
			val_acc, val_loss = val(model, val_loader)
			test_acc, test_loss = test(model,test_loader)
			final_val_acc = val_acc
			last_lr=scheduler.get_last_lr()[-1]
			scheduler.step()

		# save the model after training
		torch.save(model, save_path+'/model_save/'+ name_model)
		model_path = save_path+'/model_save/'+ name_model

		data = {'Lay.rem': Iden_num}
		data["train_acc"] = train_acc
		data["val_acc"] = val_acc
		data["test_acc"] = test_acc
		accumulated_data = accumulated_data.append(data, ignore_index=True)
		accumulated_data.to_excel(save_path+'result.xlsx', index=False)

		# count the number of remaining ReLU layers
		relu_num = 0
		for name, module in model.named_modules():
			if type(module) == torch.nn.ReLU6:
				relu_num +=1

		# stop the loop once no ReLU layer left or model no longer perform well
		if relu_num ==1 or final_val_acc < 20:
			break

		wandb.finish()


if __name__ == '__main__':
	main()