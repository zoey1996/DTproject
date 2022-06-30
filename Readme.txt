"""The code of this project is written based on https://github.com/luoyunan/DTINet and https://github.com/devalab/molgpt, and I would like to express my heartfelt thanks for their open source code!"""
1.Environment configuration
	conda env create -f DTproject_environment.yml
	
	Additional installed packages:
	
		pybiomed == 1.0:https://github.com/gadsbyfly/PyBioMed/
		torch==1.8.1+cu101:pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

	Add environment to ipykernel:
	
		pip install ipykernel ipython
		ipython kernel install --user --name DTproject

2.Code
	DT_predict contains files and code files related to transporter prediction:
		The KG+seq_featrue_train.ipynb uses the KG embedding+seq feature as the input feature to train the model code.
		The Seq_featrue_train.ipynb is the code to train the model with only seq features as input features.
		The predicated_task.ipynb is the code for luteolin instance verification.
		The KG_embedding.ipynb script in the data folder is the code for KG embedding.
		The Newdrug2kg.ipynb in the data folder is the code for adding drugs that are not included in the knowledge graph.
	
	
	DT_predict contains files and code files related to the generation of small molecules based on transporters, which are divided into two folders: training and generation.
		seq.ipynb and seq+KG.ipynb are two types of training and generation tasks based on sequence only and based on sequence and KG embedded features, respectively.