
import torch
import numpy as np
from tqdm import tqdm
import random
import wandb
import os
import re
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset, DatasetDict
import argparse



# os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda:0')  # Device configuration


class Hook():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module.register_forward_hook(self.hook_fn)                        
		else:
			self.hook = module.register_backward_hook(self.hook_fn)													 
	def hook_fn(self, module, input, output):
		self.output = input[0]  
	def close(self):
		self.hook.remove()





def cal_Pbn_entropy(model,hooks,val_loader):
	cdf_0 = {}	
	beta = {}
	gamma_bn = {}
	entropy={}	
	for key in hooks.keys(): 
		cdf_0[key] = 0
		entropy[key] = 0
		beta[key] = 0
		gamma_bn[key] = 0
	for data in tqdm(val_loader):
		inputs = {'input_ids': data['input_ids'].to(device),
					'attention_mask': data['attention_mask'].to(device),
					'labels': data['label'].to(device)}
		outputs = model(**inputs)
		for key in hooks.keys():         # For different layers	
			Beta = hooks[key].output					
			while len(Beta.shape) > 1:					
				Beta = torch.mean(Beta,dim=0)
			Gamma = torch.std(hooks[key].output, dim=[0,1])
			normal_part = 0.5 * (1 + torch.erf((0 - Beta) / (Gamma * np.sqrt(2))))
			zero_gamma_condition = Gamma == 0
			cdf_current= torch.where(zero_gamma_condition, (0 >= Beta).float(), normal_part)
			entropy_current = -torch.mean(cdf_current* torch.log2(torch.clamp(cdf_current, min=1e-5)) + (1 - cdf_current) * torch.log2(torch.clamp(1 - cdf_current, min=1e-5)))
			cdf_0[key] += cdf_current
			entropy[key] += entropy_current
			beta[key] += Beta
			gamma_bn[key] += Gamma
		break
	for key in hooks.keys():
		cdf_0[key] = cdf_0[key] 
		entropy[key] = entropy[key] 
		beta[key] = beta[key]  
		gamma_bn[key] = gamma_bn[key] 
	return (cdf_0, beta, gamma_bn, entropy)


def train(model, epoch, optimizer,train_loader):
	print(f'\nEpoch: {epoch + 1}')
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	for data in tqdm(train_loader):
		inputs = {'input_ids': data['input_ids'].to(device),
					'attention_mask': data['attention_mask'].to(device),
					'labels': data['label'].to(device)}
		outputs = model(**inputs)
		loss = outputs.loss
		predictions = outputs.logits.argmax(dim=1)    
		total_loss = loss        
		total += inputs['labels'].size(0)
		correct += (predictions == inputs['labels']).sum().item()		
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
	with torch.no_grad():
		for data in tqdm(val_loader):
			inputs = {'input_ids': data['input_ids'].to(device),
						'attention_mask': data['attention_mask'].to(device),
						'labels': data['label'].to(device)}
			outputs = model(**inputs)
			loss = outputs.loss
			predictions = outputs.logits.argmax(dim=1)
			running_loss += loss.item()
			correct += (predictions == inputs['labels']).sum().item() 
			total += inputs['labels'].size(0)	
	test_loss=running_loss/len(val_loader)
	accuracy=100.*correct/total

	print(f'Val Loss: {test_loss:.3f} | Accuracy: {accuracy:.3f}')
	return(accuracy, test_loss)


def test(model,test_loader):
	model.eval()
	running_loss=0
	correct=0
	total=0    
	with torch.no_grad():
		for data in tqdm(test_loader):
			inputs = {'input_ids': data['input_ids'].to(device),
						'attention_mask': data['attention_mask'].to(device),
						'labels': data['label'].to(device)}
			outputs = model(**inputs)
			loss = outputs.loss
			predictions = outputs.logits.argmax(dim=1)
			running_loss += loss.item()
			correct += (predictions == inputs['labels']).sum().item() 	
			total += inputs['labels'].size(0)
	test_loss=running_loss/len(test_loader)
	accuracy=100.*correct/total

	print(f'Test Loss: {test_loss:.3f} | Accuracy: {accuracy:.3f}')
	return(accuracy, test_loss)



def testVal(model,val_loader):
	model.eval()	
	running_loss=0
	correct=0
	total=0    
	with torch.no_grad():
		for data in tqdm(val_loader):
			inputs = {'input_ids': data['input_ids'].to(device),
						'attention_mask': data['attention_mask'].to(device),
						'labels': data['label'].to(device)}
			outputs = model(**inputs)
			loss = outputs.loss
			predictions = outputs.logits.argmax(dim=1)
			running_loss += loss.item()
			correct += (predictions == inputs['labels']).sum().item() 	
			total += inputs['labels'].size(0)
			break
	test_loss=running_loss
	accu=100.*correct/total
	print('testVal_loss Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
	return(accu, test_loss)



def replace_gelu_activation(model):
	for name, module in model.named_children():
		if 'intermediate_act_fn' in name:
			setattr(model, name, torch.nn.GELU())
		else:
			replace_gelu_activation(module) 



def main():

	parser = argparse.ArgumentParser(description='Pbn_NLP')
	parser.add_argument('--dataset', default='sst2')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--lr', type=float, default=2e-5)
	parser.add_argument('--epochs', type=int, default=3)	
	parser.add_argument('--seed',  type=int, default=43, help='seed')	
	args = parser.parse_args()


	# set seeds
	torch.manual_seed(args.seed)
	os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.use_deterministic_algorithms(True)


	save_path = './' +'BERT/'+args.dataset+'/'
	if not os.path.exists(save_path+'/model_save/'):
		os.makedirs(save_path+'/model_save/')


	model_name = 'bert-base-uncased'
	tokenizer = BertTokenizer.from_pretrained(model_name)
	if args.dataset == 'sst2' or args.dataset == 'rte' or args.dataset == 'qnli':
		model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
	replace_gelu_activation(model)


	dataset = load_dataset('glue', args.dataset)
			
	def rte_tokenize_function(examples):
		return tokenizer(examples['sentence1'], examples['sentence2'], 
						padding="max_length", truncation=True, max_length=128)

	def qnli_tokenize_function(examples):
		return tokenizer(examples['question'], examples['sentence'], 
						padding="max_length", truncation=True, max_length=128)

	if args.dataset == 'sst2':
		dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=128), batched=True)
	elif args.dataset == 'rte' :
		dataset = dataset.map(rte_tokenize_function, batched=True)
	elif args.dataset == 'qnli':
		dataset = dataset.map(qnli_tokenize_function, batched=True)


	dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
	hooks = {}
	for name, module in model.named_modules():
		if type(module) == torch.nn.GELU:
			hooks[name] = Hook(module)

	# (9:1)
	train_test_split = dataset['train'].train_test_split(test_size=0.1)
	dataset = DatasetDict({
		'train': train_test_split['train'],
		'validation': train_test_split['test'],
		'test': dataset['validation'] 
	})
	train_loader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(dataset['validation'], batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)



	data_columns = ['Lay.rem'] + ["train_acc"] + ["val_acc"] + ["test_acc"] 
	accumulated_data = pd.DataFrame(columns=data_columns)


	for it in range(0, 40):
		if it > 0:

			cdf_0, beta, bn_gamma, entropy = cal_Pbn_entropy(model,hooks,val_loader)
			temp_layer = []
			layers_to_replace = []
			layers_to_prune=[]
			for key in cdf_0.keys():
				temp_layer.append(key)

			beta_conv = {}
			beta_bn = {}
			previous_bn = False
			for name, module in model.named_modules():
				if type(module) == torch.nn.Conv2d or type(module) == torch.nn.Linear:
					previous_conv_name = name
				elif type(module) == torch.nn.BatchNorm2d:
					previous_bn_name = name		
					previous_bn = True	
				elif name in temp_layer:
					beta_conv[previous_conv_name] = beta[name]
					if previous_bn == True:	
						beta_bn[previous_bn_name] = beta[name]
						previous_bn = False


			loss_wo_layer = {}
			acc_wo_layer = {}
			loss_gap={}
			acc_gap={}

			testVal_acc, testVal_loss = testVal(model,val_loader)
			loss_wo_layer['no']=testVal_loss
			acc_wo_layer['no']=testVal_acc

			for name_conv, module in model.named_modules():
				if name_conv in beta_conv.keys() :
					print(name_conv)
					bn_close = []
					layers_to_replace = []
					model_copy = torch.load(model_path).to(device)
					previous_conv = False
					for name, module in model_copy.named_modules():
						if name == name_conv:
							previous_conv_name = name
							previous_conv = True
						elif type(module) == torch.nn.BatchNorm2d and previous_conv == True:
							bn_close.append(name)
						elif type(module) == torch.nn.GELU and previous_conv == True:
							layers_to_replace.append(name)
							previous_conv = False

					for name, module in model_copy.named_modules():
						if name == name_conv:				
							layer_mask = []
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
							for i in range(beta_bn[name].size()[0]):
								if beta_bn[name][i] < 0 :
									module.weight.data[i] = 0                                                        
									module.bias.data[i] = 0		

					for name in layers_to_replace:  
						change_name = re.sub(r'layer\.(\d+)\.', r'layer[\1].', name)
						exec(f'model_copy.{change_name} = nn.Identity()')

					testVal_acc, testVal_loss = testVal(model_copy,val_loader)
					loss_wo_layer[name_conv]=testVal_loss
					loss_gap[name_conv] = testVal_loss - loss_wo_layer['no']
					acc_gap[name_conv] =  acc_wo_layer['no'] - testVal_acc
								
			sorted_loss_gap = dict(sorted(loss_gap.items(), key=lambda item: item[1]))




			layers_to_prune = []
			if min(sorted_loss_gap.values()) < 0:
				for num_combine in range(2,len(loss_gap)):
					combine_remove_layer = [key for key, value in sorted(sorted_loss_gap.items(), key=lambda item: item[1])][:num_combine]
					print('remove layers',combine_remove_layer)
					model_com_rem = torch.load(model_path).to(device)
					bn_close = []
					layers_to_replace = []
					previous_conv = False
					for name, module in model_com_rem.named_modules():
						if name in combine_remove_layer :
							previous_conv_name = name
							previous_conv = True
						elif type(module) == torch.nn.BatchNorm2d and previous_conv == True:
							bn_close.append(name)
						elif type(module) == torch.nn.GELU and previous_conv == True:
							layers_to_replace.append(name)
							previous_conv = False

					for name, module in model_com_rem.named_modules():
						if name in combine_remove_layer:
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

					for name, module in model_com_rem.named_modules():
						if name in bn_close:
							for i in range(beta_bn[name].size()[0]):
								if beta_bn[name][i] < 0 :
									module.weight.data[i] = 0                                                        
									module.bias.data[i] = 0		

					for name in layers_to_replace:  
						change_name = re.sub(r'layer\.(\d+)\.', r'layer[\1].', name)
						exec(f'model_com_rem.{change_name} = nn.Identity()')	

					testVal_acc, testVal_loss = testVal(model_com_rem,val_loader)

					if testVal_loss - loss_wo_layer['no'] > 0:
						layers_to_prune = combine_remove_layer
						break



			if not layers_to_prune and loss_gap:
				min_key = min(loss_gap, key=loss_gap.get)
				layers_to_prune.append(min_key)



			model = torch.load(model_path).to(device)
			bn_close = []
			layers_to_replace = []
			for name, module in model.named_modules():
				if name in layers_to_prune :
					previous_conv_name = name
					previous_conv = True
				elif type(module) == torch.nn.BatchNorm2d and previous_conv == True:
					bn_close.append(name)
				elif type(module) == torch.nn.GELU and previous_conv == True:
					layers_to_replace.append(name)
					previous_conv = False

			for name, module in model.named_modules():
				if name in layers_to_prune:
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

			for name, module in model.named_modules():
				if name in bn_close:
					for i in range(beta_bn[name].size()[0]):
						if beta_bn[name][i] < 0 :
							module.weight.data[i] = 0                                                        
							module.bias.data[i] = 0		

			for name in layers_to_replace:  
				hooks.pop(name)
				change_name = re.sub(r'layer\.(\d+)\.', r'layer[\1].', name)
				exec(f'model.{change_name} = nn.Identity()')	

		Iden_num = 0
		Iden_list = []
		for name, module in model.named_modules():
			if type(module) == torch.nn.Identity:
				Iden_num +=1
				Iden_list.append(name)


		val_acc, val_loss = val(model,val_loader)
		test_acc, test_loss = test(model,test_loader)





		name_of_run =  "it_"+str(it) + '_Iden_' + str(Iden_num)
		name_model = name_of_run

		optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
		for epoch in range(args.epochs):
			train_acc, train_loss = train(model, epoch, optimizer,train_loader)
			val_acc, val_loss = val(model,val_loader)
			test_acc, test_loss = test(model,test_loader)
			final_val_acc = val_acc
		torch.save(model, save_path+'model_save/'+ name_model)
		model_path = save_path+'model_save/'+ name_model


		data = {'Lay.rem': Iden_num}
		data["train_acc"] = train_acc
		data["val_acc"] = val_acc
		data["test_acc"] = test_acc

		accumulated_data = accumulated_data.append(data, ignore_index=True)
		accumulated_data.to_excel(save_path+'result.xlsx', index=False)


		relu_num = 0
		for name, module in model.named_modules():
			if type(module) == torch.nn.GELU:
				relu_num +=1

		if relu_num ==1 or final_val_acc < 20:
			break

		wandb.finish()
####################################################################################################################


if __name__ == '__main__':
	main()