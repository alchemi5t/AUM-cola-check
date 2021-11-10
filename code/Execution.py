#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     AUM
# @Filename:    Execution.py
# @Author:      mdagar
# @Time:        4/12/21 4:14 PM

import torch
from aum import AUMCalculator
from tqdm import tqdm
import os

class Execution:
    @staticmethod
    def evaluation(model, test_data, bsize):
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=bsize, shuffle=True)
            model.eval()
            acc = 0.0
            total_size = 0
            for batch in test_loader:
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
                outputs = model(input_ids, attention_mask, labels=labels)
                acc += torch.sum(outputs.logits.argmax(dim=-1) == labels)
                total_size += bsize
            print("Test Acc: ", acc / total_size)
    
    @staticmethod
    def training(model, bsize, n_epochs, train_data, test_data , lrate, optimizer, with_aum, aum_path=None, aum_calculator=None,evaluate = False):
        def evaluation(model, test_data, bsize):
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=bsize, shuffle=True)
            model.eval()
            acc = 0.0
            total_size = 0
            for batch in test_loader:
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
                outputs = model(input_ids, attention_mask, labels=labels)
                acc += torch.sum(outputs.logits.argmax(dim=1) == labels)
                total_size += bsize
            print("Test Acc: ", acc / total_size)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=bsize, shuffle=True)
        model.cuda()
        opt = optimizer(params=model.parameters(), lr=lrate)
        model.train()
        for i in (range(n_epochs)):
            total_train_loss = 0.0
            acc = 0.0
            total_size = 0
            for batch in tqdm(dataloader, leave=False):
                label = batch["labels"].cuda()
                data = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()

                outputs = model(data, attention_mask=attention_mask, labels=label)
                acc += torch.sum(outputs.logits.argmax(dim=1) == label)
                total_size += bsize
                loss = outputs.loss

                if with_aum:
                    index = batch["index"]
                    #lb = batch["labels"]
                    #logits = outputs.logits.cpu()
                    records = aum_calculator.update(outputs.logits, label, index.tolist())

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_train_loss += loss
            if(evaluate):
              evaluation(model, test_data, bsize)
            print("Train_Loss:", total_train_loss / len(dataloader), "Acc: ", acc / total_size)

        if with_aum:
            if os.path.exists(aum_path+"aum_values.csv"):
                os.remove(aum_path+"aum_values.csv")
            aum_calculator.finalize()

        return model

    
