import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pyepo.model.opt import optModel

class cvrpTextDataset(Dataset):
    def __init__(self, csv_file: str, opt_model: optModel, tokenizer_name: str = "microsoft/deberta-v3-base", max_length: int = 128):
        self.data = csv_file
        self.opt_model = opt_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # precomputing w*(c) and z*(c)
        self.w_star = []
        self.z_star = []
        print(f"Precomputing of w*(c) and z*(c) for all instances with {opt_model} ...")
        for i in range(len(self.data)):
            c = np.array(json.loads(self.data.iloc[i]["costs"]))
            self.opt_model.setObj(c)
            sol, obj = self.opt_model.solve()
            self.w_star.append(torch.tensor(sol, dtype=torch.float32))
            self.z_star.append(torch.tensor(obj, dtype=torch.float32))
            if (i + 1) % 100 == 0:
                print(f"Resolution : {i + 1} / {len(self.data)} CVRP solved.")
        print("Precomputing finished !")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx: int) -> dict:
            row = self.data.iloc[idx]
            node_texts = json.loads(row["node_texts"])
            costs = torch.tensor(json.loads(row["costs"]), dtype=torch.float32)
            encoded = self.tokenizer(
                node_texts,
                padding=max_length,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
            "input_ids": encoded["input_ids"],          
            "attention_mask": encoded["attention_mask"],
            "costs": costs,
            "w_star": self.w_stars[idx],
            "z_star": self.z_stars[idx]
        }
