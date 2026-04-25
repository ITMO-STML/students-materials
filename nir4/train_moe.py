#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import torch
import json
import numpy as np
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_from_disk, Dataset as HFDataset, DatasetDict
import ir_measures
from ir_measures import nDCG, MRR, P, Recall, Qrel, ScoredDoc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from modeling_xlm_roberta_moe import XLMRobertaMoEConfig, XLMRobertaMoEModel, XLMRobertaMoEForMaskedLM

from clearml import Task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    data_config: Dict[str, List[str]]
    output_dir: str = "./moe_training"
    model_config_path: str = None
    resume_from_checkpoint: Optional[str] = None
    batch_size: int = 8
    grad_acc_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 1000
    max_steps: int = 100000
    max_epochs: Optional[int] = None
    save_by_steps: bool = True
    save_steps: int = 5000
    save_epochs: float = 1.0
    eval_by_steps: bool = True
    eval_steps: int = 1000
    eval_epochs: float = 1.0
    logging_steps: int = 100
    max_length: int = 512
    seed: int = 42
    num_workers: int = 4
    
    log_to_clearml: bool = False
    clearml_tags: List[str] = None
    task_id: str = ""
    project_name: str = "Semantic Search"
    experiment_name: str = None

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def setup_clearml(config):
    if not config.log_to_clearml:
        logger.info("ClearML logging disabled")
        return None
    
    logger.info("=" * 60)
    logger.info("ClearML initialization")
    logger.info("=" * 60)
    
    if config.task_id:
        task = Task.init(
            reuse_last_task_id=config.task_id,
            continue_last_task=0,
            task_type="training",
        )
        logger.info(f"Reusing task: {task.id}")
    else:
        exp_name = config.experiment_name or f"moe_training"
        task = Task.init(
            project_name=config.project_name,
            task_name=exp_name,
            task_type="training",
        )
        logger.info(f"Created new task: {task.id}")
    
    logger.info(f"  Project: {task.get_project_name()}")
    logger.info(f"  Name: {task.name}")
    logger.info(f"  Web URL: {task.get_output_log_web_page()}")
    
    if config.clearml_tags:
        task.add_tags(config.clearml_tags)
        logger.info(f"  Tags: {config.clearml_tags}")
    
    task.connect(asdict(config))
    logger.info("=" * 60)
    
    return task


def find_best_path(base_path, subfolder):
    versions = ["_2.0.0", "_1.0.0", ""]
    
    for version in versions:
        candidate = os.path.join(base_path, f"{subfolder}{version}")
        if os.path.exists(candidate):
            logger.info(f"Found {subfolder} at: {candidate}")
            return candidate
    
    raise FileNotFoundError(f"Cannot find {subfolder} in {base_path}")


class RetrievalDatasetFromFolder(Dataset):
    
    def __init__(self, base_path: str, split: str, lang: str):
        self.lang = lang
        self.split = split
        
        corpus_path = find_best_path(base_path, "corpus")
        dataset_path = find_best_path(base_path, "dataset")
        
        logger.info(f"Loading {split} dataset from folder for language {lang}")
        logger.info(f"  Corpus: {corpus_path}")
        logger.info(f"  Dataset: {dataset_path}")
        
        self.corpus = load_from_disk(corpus_path)
        self.corpus_dict = {
            str(id): text for id, text in zip(self.corpus['id'], self.corpus['text'])
        }
        
        dataset_dict = load_from_disk(dataset_path)
        if isinstance(dataset_dict, DatasetDict):
            self.queries = dataset_dict[split]
        else:
            self.queries = dataset_dict
        
        self.has_neg = 'negatives' in self.queries.column_names if len(self.queries) > 0 else False
        
        logger.info(f"  {len(self.queries)} queries, {len(self.corpus_dict)} docs")
        logger.info(f"  has negatives: {self.has_neg}")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_item = self.queries[idx]
        
        pos_doc_ids = [str(id) for id in query_item['positives']['doc_id']]
        pos_docs = [self.corpus_dict[id] for id in pos_doc_ids]
        
        result = {
            "query": query_item['query'],
            "pos_docs": pos_docs,
            "pos_doc_ids": pos_doc_ids,
            "lang": self.lang,
            "has_neg": self.has_neg,
            "idx": idx,
            "split": self.split
        }
        
        if self.has_neg:
            neg_doc_ids = [str(id) for id in query_item['negatives']['doc_id']]
            neg_docs = [self.corpus_dict[id] for id in neg_doc_ids]
            result["neg_docs"] = neg_docs
            result["neg_doc_ids"] = neg_doc_ids
        
        return result


class RetrievalDatasetFromJSON(Dataset):
    
    def __init__(self, json_path: str, lang: str, split: str = "train"):
        self.lang = lang
        self.split = split
        logger.info(f"Loading {split} dataset from JSON for language {lang}: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.has_neg = 'neg' in self.data[0] if self.data else False
        
        logger.info(f"  {len(self.data)} examples")
        logger.info(f"  has negatives: {self.has_neg}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        result = {
            "query": item['query'],
            "pos_docs": [item['pos']],
            "lang": self.lang,
            "has_neg": self.has_neg,
            "idx": idx,
            "split": self.split
        }
        
        if self.has_neg:
            result["neg_docs"] = [item['neg']]
        
        return result


class MixedDataset(Dataset):
    
    def __init__(self, config: TrainingConfig, split: str = "train"):
        self.config = config
        self.split = split
        
        self.items = []
        self.metadata = []
        
        for lang, paths in config.data_config.items():
            for path_idx, path in enumerate(paths):
                try:
                    if path.endswith('.json'):
                        ds = RetrievalDatasetFromJSON(path, lang, split)
                        name = f"{lang}_json_{path_idx}"
                    else:
                        ds = RetrievalDatasetFromFolder(path, split, lang)
                        name = f"{lang}_folder_{path_idx}"
                    
                    for idx in range(len(ds)):
                        item = ds[idx]
                        self.items.append(item)
                        self.metadata.append({
                            'lang': lang,
                            'has_neg': ds.has_neg,
                            'dataset_idx': len(self.metadata),
                            'name': name
                        })
                    
                    logger.info(f"Added {split} dataset {name}: {len(ds)} examples, has_neg={ds.has_neg}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        if len(self.items) == 0:
            raise ValueError(f"No datasets loaded for split '{split}'")
        
        logger.info(f"Total examples: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx].copy()
        meta = self.metadata[idx]
        
        item['lang'] = meta['lang']
        item['has_neg'] = meta['has_neg']
        item['dataset_idx'] = meta['dataset_idx']
        
        return item


class MoECollator:
    
    def __init__(self, tokenizer, lang_to_idx, num_layers, max_length=512):
        self.tokenizer = tokenizer
        self.lang_to_idx = lang_to_idx
        self.num_layers = num_layers
        self.max_length = max_length

    def _create_expert_paths(self, encodings, lang_idx, batch_size):
        seq_len = encodings['input_ids'].size(1)
        expert_paths = torch.zeros(batch_size, seq_len, self.num_layers, dtype=torch.long)
        
        for i in range(batch_size):
            real_len = int(encodings['attention_mask'][i].sum().item())
            expert_paths[i, :real_len, :] = lang_idx
        
        return expert_paths

    def _tokenize(self, texts, prefix):
        return self.tokenizer(
            [f"{prefix}: {t}" for t in texts],
            padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )

    def _process_group(self, group_items):
        langs = [item['lang'] for item in group_items]
        assert len(set(langs)) == 1, f"Mixed languages in group: {langs}"
        
        lang = langs[0]
        lang_idx = self.lang_to_idx[lang]
        has_neg = group_items[0]['has_neg']
        batch_size = len(group_items)
        
        queries = [item["query"] for item in group_items]
        q_enc = self._tokenize(queries, "query")
        
        pos_docs = [item["pos_docs"][0] for item in group_items]
        pos_enc = self._tokenize(pos_docs, "passage")
        
        q_expert_paths = self._create_expert_paths(q_enc, lang_idx, batch_size)
        pos_expert_paths = self._create_expert_paths(pos_enc, lang_idx, batch_size)
        
        result = {
            "query_input_ids": q_enc["input_ids"],
            "query_attention_mask": q_enc["attention_mask"],
            "query_expert_paths": q_expert_paths,
            "pos_doc_input_ids": pos_enc["input_ids"],
            "pos_doc_attention_mask": pos_enc["attention_mask"],
            "pos_doc_expert_paths": pos_expert_paths,
            "has_neg": has_neg,
            "lang": lang,
            "indices": [item.get('idx', i) for i, item in enumerate(group_items)]
        }
        
        if has_neg:
            neg_docs = [item["neg_docs"][0] for item in group_items]
            neg_enc = self._tokenize(neg_docs, "passage")
            neg_expert_paths = self._create_expert_paths(neg_enc, lang_idx, batch_size)
            
            result["neg_doc_input_ids"] = neg_enc["input_ids"]
            result["neg_doc_attention_mask"] = neg_enc["attention_mask"]
            result["neg_doc_expert_paths"] = neg_expert_paths
        
        return result

    def __call__(self, batch_items):
        groups = {}
        for item in batch_items:
            ds_idx = item['dataset_idx']
            if ds_idx not in groups:
                groups[ds_idx] = []
            groups[ds_idx].append(item)
        
        group_results = [self._process_group(group) for group in groups.values()]
        
        if len(group_results) == 1:
            return group_results[0]
        
        merged = {}
        for key in group_results[0].keys():
            if key in ['has_neg', 'lang', 'indices']:
                merged[key] = group_results[0][key]
            else:
                merged[key] = torch.cat([r[key] for r in group_results], dim=0)
        
        return merged


class MoEForRetrieval(torch.nn.Module):
    
    def __init__(self, config: XLMRobertaMoEConfig):
        super().__init__()
        self.encoder = None
        self.temperature = 0.05

    def encode(self, input_ids, attention_mask, expert_paths=None):
        if attention_mask is not None and attention_mask.dtype == torch.long:
            attention_mask = attention_mask.float()
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expert_paths=expert_paths
        )
        return self._pool(outputs.last_hidden_state, attention_mask)

    def _pool(self, last_hidden, mask):
        if mask.dtype == torch.long:
            mask = mask.float()
        mask = mask.unsqueeze(-1)
        last_hidden = last_hidden.masked_fill(~mask.bool(), 0.0)
        emb = last_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, query_batch, pos_doc_batch, neg_doc_batch=None):
        q_emb = self.encode(
            query_batch["input_ids"],
            query_batch["attention_mask"],
            query_batch.get("expert_paths")
        )
        
        pos_emb = self.encode(
            pos_doc_batch["input_ids"],
            pos_doc_batch["attention_mask"],
            pos_doc_batch.get("expert_paths")
        )
        
        if neg_doc_batch is not None:
            neg_emb = self.encode(
                neg_doc_batch["input_ids"],
                neg_doc_batch["attention_mask"],
                neg_doc_batch.get("expert_paths")
            )
            return q_emb, pos_emb, neg_emb
        else:
            return q_emb, pos_emb

    def compute_loss(self, q_emb, pos_emb, neg_emb=None):
        batch_size = q_emb.size(0)
        
        q_emb = F.normalize(q_emb, p=2, dim=1)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)
        
        pos_scores = torch.sum(q_emb * pos_emb, dim=1, keepdim=True) / self.temperature
        in_batch_scores = torch.matmul(q_emb, pos_emb.T) / self.temperature
        
        mask = torch.eye(batch_size, device=q_emb.device).bool()
        in_batch_scores.masked_fill_(mask, -float('inf'))
        
        if neg_emb is not None:
            neg_emb = F.normalize(neg_emb, p=2, dim=1)
            hard_neg_scores = torch.matmul(q_emb, neg_emb.T) / self.temperature
            all_scores = torch.cat([pos_scores, in_batch_scores, hard_neg_scores], dim=1)
        else:
            all_scores = torch.cat([pos_scores, in_batch_scores], dim=1)
        
        logsumexp = torch.logsumexp(all_scores, dim=1)
        loss = (-pos_scores.squeeze() + logsumexp).mean()
        
        return loss
    
    def to(self, device):
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
        return self


def compute_retrieval_metrics(q_embs, doc_embs, q_ids, doc_ids, qrels_dict):
    qrels = []
    run = []
    
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    
    for qid, rel_docs in qrels_dict.items():
        for doc_id in rel_docs:
            if doc_id in doc_id_to_idx:
                qrels.append(Qrel(qid, doc_id, 1))
    
    if not qrels:
        return {"nDCG@10": 0.0, "MRR@10": 0.0, "P@10": 0.0, "R@10": 0.0}
    
    q_embs = F.normalize(torch.from_numpy(q_embs), p=2, dim=1).numpy()
    doc_embs = F.normalize(torch.from_numpy(doc_embs), p=2, dim=1).numpy()
    
    scores = np.dot(q_embs, doc_embs.T)
    
    for qid_idx, (qid, rel_docs) in enumerate(qrels_dict.items()):
        q_scores = scores[qid_idx]
        sorted_indices = np.argsort(q_scores)[::-1]
        for rank, doc_idx in enumerate(sorted_indices[:100]):
            doc_id = doc_ids[doc_idx]
            run.append(ScoredDoc(qid, doc_id, float(q_scores[doc_idx])))
    
    metrics = ir_measures.calc_aggregate([nDCG@10, MRR@10, P@10, Recall@10], qrels, run)
    
    return {
        "nDCG@10": float(metrics[nDCG@10]),
        "MRR@10": float(metrics[MRR@10]),
        "P@10": float(metrics[P@10]),
        "R@10": float(metrics[Recall@10])
    }


def evaluate_model_on_datasets(model, datasets, tokenizer, config, split_name="test", device="cuda"):
    model.eval()
    all_metrics = {}
    
    with torch.no_grad():
        for ds_idx, dataset in enumerate(datasets):
            logger.info(f"Evaluating {split_name} {dataset.lang} dataset {ds_idx}...")
            
            if split_name == "val" and len(dataset) > 2000:
                max_eval = 2000
                indices = random.sample(range(len(dataset)), max_eval)
                logger.warning(f"  Dataset too large ({len(dataset)} examples), using random sample of {max_eval}")
            else:
                max_eval = len(dataset)
                indices = range(max_eval)
            
            queries = []
            all_docs = []
            qrels_dict = {}
            
            for i in indices:
                item = dataset[i]
                queries.append(item['query'])
                
                for pos_doc in item['pos_docs']:
                    if pos_doc not in all_docs:
                        all_docs.append(pos_doc)
                
                doc_indices = [all_docs.index(doc) for doc in item['pos_docs']]
                qrels_dict[f"q_{ds_idx}_{i}"] = [f"doc_{idx}" for idx in doc_indices]
            
            q_embs = []
            for i in range(0, len(queries), config.batch_size):
                batch_q = queries[i:i+config.batch_size]
                q_inputs = tokenizer([f"query: {q}" for q in batch_q], 
                                    padding=True, truncation=True,
                                    max_length=config.max_length,
                                    return_tensors="pt").to(device)
                emb = model.encode(q_inputs["input_ids"], q_inputs["attention_mask"])
                q_embs.append(emb.cpu().numpy())
            q_embs = np.vstack(q_embs)
            
            doc_embs = []
            doc_ids = []
            for i in range(0, len(all_docs), config.batch_size):
                batch_d = all_docs[i:i+config.batch_size]
                d_inputs = tokenizer([f"passage: {d}" for d in batch_d], 
                                    padding=True, truncation=True,
                                    max_length=config.max_length,
                                    return_tensors="pt").to(device)
                emb = model.encode(d_inputs["input_ids"], d_inputs["attention_mask"])
                doc_embs.append(emb.cpu().numpy())
                
                for j in range(i, i+len(batch_d)):
                    doc_ids.append(f"doc_{j}")
            
            doc_embs = np.vstack(doc_embs)
            
            metrics = compute_retrieval_metrics(
                q_embs, doc_embs,
                list(qrels_dict.keys()), doc_ids,
                qrels_dict
            )
            
            dataset_name = f"{split_name}_{dataset.lang}_{ds_idx}"
            all_metrics[dataset_name] = metrics
            
            logger.info(f"  {dataset_name}: nDCG@10={metrics['nDCG@10']:.4f}")
    
    return all_metrics


class MoETrainer:
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        self.clearml_task = setup_clearml(config)

        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        
        if not self.config.model_config_path:
            raise ValueError("model_config_path must be provided")
        
        logger.info(f"Loading model config from {self.config.model_config_path}")
        model_config = XLMRobertaMoEConfig.from_pretrained(self.config.model_config_path)
        self.num_layers = model_config.num_hidden_layers
        
        self._setup_data()
        
        self._setup_model()
        
        self._check_initialization()
        
        self._compute_max_steps()
        self._setup_optimizer()

        self.global_step = 0
        self.best_metric = float('-inf')
        self.current_epoch = 0

    def _setup_model(self):
        logger.info(f"Loading model config from {self.config.model_config_path}")
        model_config = XLMRobertaMoEConfig.from_pretrained(self.config.model_config_path)
        
        tokenizer_vocab = len(self.tokenizer)
        logger.info(f"Tokenizer vocab size: {tokenizer_vocab}")
        logger.info(f"Model config vocab size: {model_config.vocab_size}")
        
        if tokenizer_vocab != model_config.vocab_size:
            logger.warning(f"Vocab size mismatch! Using tokenizer vocab: {tokenizer_vocab}")
            model_config.vocab_size = tokenizer_vocab
        
        logger.info(f"Model config loaded: {model_config.num_hidden_layers} layers")
        logger.info(f"MoE config: {model_config.moe_num_experts}")
        
        logger.info("Loading E5-large weights and converting to MoE...")
        moe_model = XLMRobertaMoEForMaskedLM.from_dense_pretrained(
            "intfloat/multilingual-e5-large",
            moe_num_experts=model_config.moe_num_experts,
        )
        
        self.model = MoEForRetrieval(moe_model.roberta.config)
        self.model.encoder = moe_model.roberta
        self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {total_params / 1e6:.1f}M parameters")

    def _setup_data(self):
        all_langs = list(self.config.data_config.keys())
        self.lang_to_idx = {lang: i for i, lang in enumerate(sorted(all_langs))}
        logger.info(f"Language mapping: {self.lang_to_idx}")

        self.train_dataset = MixedDataset(self.config, "train")
        
        self.val_datasets = []
        self.test_datasets = []
        
        for lang, paths in self.config.data_config.items():
            for path in paths:
                if path.endswith('.json'):
                    val_ds = RetrievalDatasetFromJSON(path, lang, "val")
                    test_ds = RetrievalDatasetFromJSON(path, lang, "test")
                else:
                    val_ds = RetrievalDatasetFromFolder(path, "val", lang)
                    test_ds = RetrievalDatasetFromFolder(path, "test", lang)
                
                self.val_datasets.append(val_ds)
                self.test_datasets.append(test_ds)

        self.collator = MoECollator(
            self.tokenizer,
            self.lang_to_idx,
            num_layers=self.num_layers,
            max_length=self.config.max_length
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collator
        )

    def _check_initialization(self):
        logger.info("=" * 60)
        logger.info("Checking model initialization")
        logger.info("=" * 60)
        
        all_ok = True
        
        for layer_idx in range(len(self.model.encoder.encoder.layer)):
            moe_layer = self.model.encoder.encoder.layer[layer_idx]
            num_experts = len(moe_layer.moe.experts)
            
            logger.info(f"Layer {layer_idx}: {num_experts} experts")
            
            if num_experts > 1:
                ref_up = moe_layer.moe.experts[0].up_proj.weight.data.clone()
                ref_down = moe_layer.moe.experts[0].down_proj.weight.data.clone()
                
                if torch.all(ref_up == 0):
                    logger.error(f"  Layer {layer_idx} up_proj weights are ZERO")
                    all_ok = False
                else:
                    logger.info(f"  Layer {layer_idx} up_proj non-zero (mean={ref_up.mean():.6f})")
                
                all_equal = True
                for exp_idx in range(1, num_experts):
                    up_equal = torch.allclose(moe_layer.moe.experts[exp_idx].up_proj.weight.data, ref_up, rtol=1e-5)
                    down_equal = torch.allclose(moe_layer.moe.experts[exp_idx].down_proj.weight.data, ref_down, rtol=1e-5)
                    
                    if not up_equal or not down_equal:
                        logger.warning(f"  Expert {exp_idx} differs from expert 0")
                        all_equal = False
                
                if all_equal:
                    logger.info(f"  All experts in layer {layer_idx} identical")
                
                router = moe_layer.moe.router
                if router is not None:
                    if torch.all(router.weight == 0) and torch.all(router.bias == 0):
                        logger.info(f"  Router correctly zero-initialized")
                    else:
                        logger.warning(f"  Router not zero")
                        all_ok = False
            else:
                expert = moe_layer.moe.experts[0]
                if torch.all(expert.up_proj.weight == 0):
                    logger.error(f"  Layer {layer_idx} up_proj weights are ZERO")
                    all_ok = False
                else:
                    logger.info(f"  Layer {layer_idx} non-zero (mean={expert.up_proj.weight.mean():.6f})")
        
        if all_ok:
            logger.info("All weights initialized correctly")
        else:
            logger.error("Initialization problems detected")
        
        logger.info("=" * 60)

    def _compute_max_steps(self):
        total_examples = len(self.train_dataset)
        
        effective_batch = self.config.batch_size * self.config.grad_acc_steps
        steps_per_epoch = max(1, total_examples // effective_batch)
        
        if self.config.max_epochs is not None:
            self.max_steps = steps_per_epoch * self.config.max_epochs
            logger.info(f"Training for {self.config.max_epochs} epochs (~{self.max_steps} steps)")
        else:
            self.max_steps = self.config.max_steps
            logger.info(f"Training for {self.max_steps} steps")
        
        self.steps_per_epoch = steps_per_epoch

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.max_steps
        )

    def train_step(self, batch):
        self.model.train()
        
        q_ids = batch["query_input_ids"].to(self.device)
        q_mask = batch["query_attention_mask"].to(self.device)
        q_paths = batch["query_expert_paths"].to(self.device)
        
        pos_ids = batch["pos_doc_input_ids"].to(self.device)
        pos_mask = batch["pos_doc_attention_mask"].to(self.device)
        pos_paths = batch["pos_doc_expert_paths"].to(self.device)
        
        if batch['has_neg']:
            neg_ids = batch["neg_doc_input_ids"].to(self.device)
            neg_mask = batch["neg_doc_attention_mask"].to(self.device)
            neg_paths = batch["neg_doc_expert_paths"].to(self.device)
            
            q_emb, pos_emb, neg_emb = self.model(
                {"input_ids": q_ids, "attention_mask": q_mask, "expert_paths": q_paths},
                {"input_ids": pos_ids, "attention_mask": pos_mask, "expert_paths": pos_paths},
                {"input_ids": neg_ids, "attention_mask": neg_mask, "expert_paths": neg_paths}
            )
            loss = self.model.compute_loss(q_emb, pos_emb, neg_emb)
        else:
            q_emb, pos_emb = self.model(
                {"input_ids": q_ids, "attention_mask": q_mask, "expert_paths": q_paths},
                {"input_ids": pos_ids, "attention_mask": pos_mask, "expert_paths": pos_paths},
                None
            )
            loss = self.model.compute_loss(q_emb, pos_emb)
        
        loss.backward()
        return loss.item()

    def evaluate_retrieval(self, datasets, split_name="val"):
        return evaluate_model_on_datasets(
            self.model, datasets, self.tokenizer, self.config, 
            split_name=split_name, device=self.device
        )

    def save_checkpoint(self, path, is_best=False):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "pytorch_model.bin"))
        self.model.encoder.config.save_pretrained(path)

        trainer_state = {
            "step": self.global_step,
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(trainer_state, os.path.join(path, "trainer_state.pt"))

        with open(os.path.join(path, "training_config.json"), 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        tag = "best" if is_best else f"step_{self.global_step}"
        logger.info(f"Checkpoint saved to {path} ({tag})")

    def train(self):
        logger.info("Starting training")
        os.makedirs(self.config.output_dir, exist_ok=True)

        logger.info("\n" + "="*60)
        logger.info("INITIAL VALIDATION (STEP 0)")
        logger.info("="*60)
        
        val_metrics_0 = self.evaluate_retrieval(self.val_datasets, split_name="val")
        test_metrics_0 = self.evaluate_retrieval(self.test_datasets, split_name="test")
        
        if self.clearml_task:
            for ds_name, metrics in val_metrics_0.items():
                for metric_name, value in metrics.items():
                    self.clearml_task.get_logger().report_scalar(
                        title=f"validation/{metric_name}",
                        series=ds_name,
                        value=value,
                        iteration=0
                    )
            
            for ds_name, metrics in test_metrics_0.items():
                for metric_name, value in metrics.items():
                    self.clearml_task.get_logger().report_scalar(
                        title=f"test/{metric_name}",
                        series=ds_name,
                        value=value,
                        iteration=0
                    )
        
        train_iter = iter(self.train_dataloader)
        total_loss = 0

        self.optimizer.zero_grad()
        pbar = tqdm(total=self.max_steps, initial=self.global_step)

        while self.global_step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.current_epoch += 1
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            loss = self.train_step(batch)
            total_loss += loss

            if (self.global_step + 1) % self.config.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1
            pbar.update(1)

            if self.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / self.config.logging_steps
                pbar.set_postfix(loss=f"{avg_loss:.4f}", epoch=self.current_epoch)
                
                if self.clearml_task:
                    self.clearml_task.get_logger().report_scalar(
                        title="Training",
                        series="loss",
                        value=avg_loss,
                        iteration=self.global_step
                    )
                
                with open(os.path.join(self.config.output_dir, "train.log"), 'a') as f:
                    f.write(f"{self.global_step}\t{avg_loss:.4f}\t{self.current_epoch}\n")
                total_loss = 0

            if self.global_step % self.config.eval_steps == 0:
                val_metrics = self.evaluate_retrieval(self.val_datasets, split_name="val")
                avg_ndcg = np.mean([m['nDCG@10'] for m in val_metrics.values()])
                logger.info(f"Step {self.global_step}, avg val nDCG@10: {avg_ndcg:.4f}")
                
                if self.clearml_task:
                    for ds_name, metrics in val_metrics.items():
                        for metric_name, value in metrics.items():
                            self.clearml_task.get_logger().report_scalar(
                                title=f"validation/{metric_name}",
                                series=ds_name,
                                value=value,
                                iteration=self.global_step
                            )

                if avg_ndcg > self.best_metric:
                    self.best_metric = avg_ndcg
                    self.save_checkpoint(os.path.join(self.config.output_dir, "best"), is_best=True)

            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(os.path.join(self.config.output_dir, f"step_{self.global_step}"))

        pbar.close()
        self.save_checkpoint(os.path.join(self.config.output_dir, "final"))
        
        logger.info("=" * 60)
        logger.info("Final test evaluation")
        logger.info("=" * 60)
        
        test_metrics = self.evaluate_retrieval(self.test_datasets, split_name="test")
        
        out_path = os.path.join(self.config.output_dir, "test_metrics.json")
        with open(out_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test metrics saved to {out_path}")
        
        if self.clearml_task:
            for ds_name, metrics in test_metrics.items():
                for metric_name, value in metrics.items():
                    self.clearml_task.get_logger().report_scalar(
                        title=f"test/{metric_name}",
                        series=ds_name,
                        value=value,
                        iteration=self.global_step
                    )
        
        logger.info("Training completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_config', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    
    parser.add_argument("--log_to_clearml", action="store_true", default=False)
    parser.add_argument("--clearml_tags", type=str, nargs="+", default=[])
    parser.add_argument("--task_id", type=str, default="")
    parser.add_argument("--project_name", type=str, default="Semantic Search")
    parser.add_argument("--experiment_name", type=str, default=None)
    args = parser.parse_args()
    
    config = TrainingConfig.from_json(args.config)

    if args.model_config:
        config.model_config_path = args.model_config
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.max_epochs:
        config.max_epochs = args.max_epochs
    if args.resume:
        config.resume_from_checkpoint = args.resume
    
    config.log_to_clearml = args.log_to_clearml
    config.clearml_tags = args.clearml_tags
    config.task_id = args.task_id
    config.project_name = args.project_name
    config.experiment_name = args.experiment_name

    trainer = MoETrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()