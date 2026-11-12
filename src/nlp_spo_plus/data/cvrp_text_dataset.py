import os
import json
import ast
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pyepo.model.opt import optModel

class cvrpTextDataset(Dataset):
    def __init__(self, csv_file: str, opt_model: optModel, tokenizer_name: str = "microsoft/deberta-v3-base", max_length: int = 128):
        # clean the dataframe
        self.data = pd.read_csv(csv_file).dropna().reset_index(drop=True)

        self.opt_model = opt_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # dynamic cache path
        cache_path = str(csv_file).replace(".csv", f"_{self.opt_model.__class__.__name__}_cache.pt")

        if os.path.exists(cache_path):
            print(f"Loading precomputed w*(c) and z*(c) from cache: {cache_path}")
            self.w_star, self.z_star = torch.load(cache_path, weights_only=False)

            # if the cache was partially saved before a crash, resume it
            if len(self.w_star) < len(self.data):
                print(f"Cache is incomplete ({len(self.w_star)}/{len(self.data)}). Resuming Gurobi computation...")
                self._compute_and_save_cache(cache_path, start_idx=len(self.w_star))
        else:
            self.w_star = []
            self.z_star = []
            print(f"Precomputing w*(c) and z*(c) for {len(self.data)} instances... (Progress is saved every 100 instances)")
            self._compute_and_save_cache(cache_path, start_idx=0)

    def _parse_list_string(self, string_data):
        """Robustly parses stringified lists from CSVs to avoid JSON crashes"""
        try:
            return json.loads(string_data)
        except json.JSONDecodeError:
            return ast.literal_eval(string_data)

    def _compute_and_save_cache(self, cache_path, start_idx=0):
        # mute gurobi
        if hasattr(self.opt_model, '_model'):
            self.opt_model._model.Params.OutputFlag = 0

        for i in range(start_idx, len(self.data)):
            try:
                # safely parse the costs using the robust parser
                c_raw = self._parse_list_string(self.data.iloc[i]["costs"])
                c = np.array(c_raw)

                self.opt_model.setObj(c)
                sol, obj = self.opt_model.solve()
                self.w_star.append(torch.tensor(sol, dtype=torch.float32))
                self.z_star.append(torch.tensor(obj, dtype=torch.float32))

                # secure the data every 100 instances
                if (i + 1) % 100 == 0:
                    torch.save((self.w_star, self.z_star), cache_path)
                    print(f"Resolution: {i + 1} / {len(self.data)} CVRP solved and secured.")

            except Exception as e:
                # if a completely corrupted row triggers a crash, save progress before dying
                print(f"\nFATAL ERROR at instance {i}: {e}")
                print("Saving current progress to cache before crashing...")
                torch.save((self.w_star, self.z_star), cache_path)
                raise e

        print("Precomputing finished! Final save to cache...")
        torch.save((self.w_star, self.z_star), cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]

        # safely parse the strings back into python lists
        node_texts = self._parse_list_string(row["node_texts"])
        costs = torch.tensor(self._parse_list_string(row["costs"]), dtype=torch.float32)

        encoded = self.tokenizer(
            node_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "costs": costs,
            "w_star": self.w_star[idx],
            "z_star": self.z_star[idx]
        }
