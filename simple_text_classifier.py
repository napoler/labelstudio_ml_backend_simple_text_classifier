from transformers import BertTokenizer,BertModel

# 
import torch
import  torch.nn as nn
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,TensorDataset
import pytorch_lightning as pl


import pickle
import os
import numpy as np
import  logging
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from label_studio_ml.model import LabelStudioMLBase
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from tkitbilstm import  BiLSTMAttention

from performer_pytorch import PerformerLM

class LitAutoCl(pl.LightningModule):
    """分类基本类 基于bilstm

    Args:
        pl ([type]): [description]
    """
    def __init__(self,vocab_size=8021, dim=10,n_hidden=10,out_num_classes=10,embedding_enabled=False,attention=True,depth=3,basemodel="PerformerLM",lr=3e-4,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        if basemodel=="bilstm":
            self.model=BiLSTMAttention(vocab_size=self.hparams.vocab_size, dim=128,n_hidden=1024,out_num_classes=out_num_classes,embedding_enabled=True,attention=True)
        elif basemodel=="PerformerLM":
            self.model= PerformerLM(
                num_tokens = self.hparams.vocab_size,
                max_seq_len = 1024,             # max sequence length
                dim = 128,                      # dimension
                depth = 3,                     # layers
                heads = 8,                      # heads
                causal = False,                 # auto-regressive or not
                nb_features = 256,              # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head
                feature_redraw_interval = 1000, # how frequently to redraw the projection matrix, the more frequent, the slower the training
                generalized_attention = False,  # defaults to softmax approximation, but can be set to True for generalized attention
                kernel_fn = nn.LeakyReLU(),          # the kernel function to be used, if generalized attention is turned on, defaults to Relu
                reversible = True,              # reversible layers, from Reformer paper
                ff_chunks = 10,                 # chunk feedforward layer, from Reformer paper
                use_scalenorm = False,          # use scale norm, from 'Transformers without Tears' paper
                use_rezero = False,             # use rezero, from 'Rezero is all you need' paper
                # tie_embedding = False,          # multiply final embeddings with token weights for logits, like gpt decoder
                ff_glu = True,                  # use GLU variant for feedforward
                emb_dropout = 0.1,              # embedding dropout
                ff_dropout = 0.1,               # feedforward dropout
                attn_dropout = 0.1,             # post-attn dropout
                local_attn_heads = 4,           # 4 heads are local attention, 4 others are global performers
                local_window_size = 256,        # window size of local attention
                rotary_position_emb = True      # use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding
            )
            self.cl=nn.Linear(self.hparams.vocab_size,self.hparams.out_num_classes)
            self.d=nn.Dropout(0.1)
            self.ac=nn.LeakyReLU()
        self.sf=nn.Softmax(-1)
        self.loss_fc=torch.nn.CrossEntropyLoss()
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        if self.hparams.basemodel=="bilstm":
            _,embedding,_ = self.model(x)
            cl=self.sf(embedding)
        elif self.hparams.basemodel=="PerformerLM":
            embedding= self.model(x)
            embedding=self.ac(embedding)
            embedding=self.d(embedding)
            embedding=self.cl(embedding)
            cl=self.sf(embedding)
            cl=cl[:, 0]
            # print?
        
        
        return cl

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        out=self(x)
        loss = self.loss_fc(out, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        out=self(x)
        loss = self.loss_fc(out, y)
        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    
    
class SimpleTextClassifier(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(SimpleTextClassifier, self).__init__(**kwargs)
        logger.info('This is a log info')
        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Choices> type
        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        assert self.info['type'] == 'Choices'

        # the model has only one textual input
        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Text'
        ## j加入词典
        self.tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_clue_tiny")
        
        
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']
        # print("self.value",self.value)
        
        #构建分类层
        # self.cl=nn.Linear(312,len(self.info['labels']))
        
        if not self.train_output:
            # If there is no trainings, define cold-started the simple TF-IDF text classifier
            # print("xx")
            
            # This is an array of <Choice> labels
            self.labels = self.info['labels']
            self.reset_model()
            # print("x111",self.labels,)
            # # make some dummy initialization
            # self.model.fit(X=self.labels, y=list(range(len(self.labels))))
            # print("x11221")
            # print('Initialized with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
            #     from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            # ))
        else:
            # otherwise load the model from the latest training results
            self.model_file = self.train_output['model_file']
            # with open(self.model_file, mode='rb') as f:
            #     self.model = pickle.load(f)
            # and use the labels from training outputs
            self.labels = self.train_output['labels']
            # 重新恢复模型
            self.reset_model()
            self.model.load_state_dict(torch.load(self.model_file))
            self.model.eval()
            
            
            
            print('Loaded from train output with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            ))
        print("初始化成功",)
    def reset_model(self):
        print("初始化成功reset_model",)
        # self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 3)), LogisticRegression(C=10, verbose=True))
        # self.model  = BertModel.from_pretrained("clue/roberta_chinese_clue_tiny")
        self.model=LitAutoCl(out_num_classes=len(self.labels))
        
    def predict(self, tasks, **kwargs):
        print("运行predict",)
        # collect input texts
        input_texts = []
        for task in tasks:
            input_texts.append(task['data'][self.value])


        inp=self.tokenizer(input_texts,return_tensors="pt",max_length=1024,padding="max_length",truncation=True)
        # self.model.fit(0bbbbbbbbbbbbbbbbbbbbbb  , output_labels_idx)
        # mydataset=TensorDataset(inp["input_ids"],torch.Tensor(output_labels_idx).long())
        probabilities=self.model(inp["input_ids"])
        # get model predictions
        # probabilities = self.model.predict_proba(input_texts)
        
        predicted_label_indices = torch.argmax(probabilities, axis=-1).tolist()
        
        print("predicted_label_indices预测结果",predicted_label_indices)
        predicted_scores = probabilities[np.arange(len(predicted_label_indices)), predicted_label_indices]
        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = self.labels[idx]
            # prediction result for the single task
            
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]
            print("result",{'result': result, 'score': score.item()})

            # expand predictions with their scores for all tasks
            predictions.append({'result': result, 'score': score.item()})

        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        print("运行fit",)
        
        # 文本列表
        input_texts = []
        output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(self.labels)}
        
        for completion in completions:
            # get input text from task data
            # print(completion)
            if completion['annotations'][0].get('skipped') or completion['annotations'][0].get('was_cancelled'):
                continue

            input_text = completion['data'][self.value]
            # 主动分词后用空格分割开
            input_text=" ".join(self.tokenizer.tokenize("input_text"))
            # print("input_text",input_text)
            

            # get an annotation
            try:
                output_label = completion['annotations'][0]['result'][0]['value']['choices'][0]
                # print("output_label",output_label)
                output_labels.append(output_label)
                output_label_idx = label2idx[output_label]
                input_texts.append(input_text)
                output_labels_idx.append(output_label_idx)
            except :
                pass

        new_labels = set(output_labels)
        if len(new_labels) != len(self.labels):
            self.labels = list(sorted(new_labels))
            print('Label set has been changed:' + str(self.labels))
            label2idx = {l: i for i, l in enumerate(self.labels)}
            output_labels_idx = [label2idx[label] for label in output_labels]

        # train the model
        self.reset_model()
        
        inp=self.tokenizer(input_texts,return_tensors="pt",max_length=1024,padding="max_length",truncation=True)
        # self.model.fit(0bbbbbbbbbbbbbbbbbbbbbb  , output_labels_idx)
        mydataset=TensorDataset(inp["input_ids"],torch.Tensor(output_labels_idx).long())
        alen=len(mydataset)
        trainlen=int(alen*0.8)
        trainDataset,testDataset=random_split(mydataset, [trainlen,alen-trainlen])
        train_loader=DataLoader(trainDataset,batch_size=16,shuffle=True)
        test_loader=DataLoader(testDataset,batch_size=16)
        
        trainer = pl.Trainer(max_epochs=3)
        trainer.fit(self.model, train_loader,test_loader)
        model_file = os.path.join(workdir, 'model.pkl')
        torch.save(self.model.state_dict(),model_file)
        # save output resources
        
        # with open(model_file, mode='wb') as fout:
        #     pickle.dump(self.model, fout)

        train_output = {
            'labels': self.labels,
            'model_file': model_file
        }
        return train_output
