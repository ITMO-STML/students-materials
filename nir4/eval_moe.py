import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import logging
from datasets import load_from_disk, DatasetDict
import os
import random
import ir_measures
from ir_measures import nDCG, MRR, P, Recall, Qrel, ScoredDoc
import json
import yaml
import argparse

from modeling_xlm_roberta_moe import XLMRobertaMoEConfig, XLMRobertaMoEForMaskedLM, XLMRobertaMoEModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_best_path(base_path, subfolder):
    versions = ["_2.0.0", "_1.0.0", ""]
    for version in versions:
        candidate = os.path.join(base_path, f"{subfolder}{version}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot find {subfolder} in {base_path}")


def load_dataset(base_path, split="val"):
    corpus_path = find_best_path(base_path, "corpus")
    dataset_path = find_best_path(base_path, "dataset")
    
    corpus = load_from_disk(corpus_path)
    corpus_dict = {str(id): text for id, text in zip(corpus['id'], corpus['text'])}
    
    dataset_dict = load_from_disk(dataset_path)
    if isinstance(dataset_dict, DatasetDict):
        queries = dataset_dict[split]
    else:
        queries = dataset_dict
    
    return queries, corpus_dict


def encode_texts_base(texts, model, tokenizer, prefix, device, batch_size=16, max_length=512):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [f"{prefix}: {t}" for t in batch]
        inputs = tokenizer(batch, padding=True, truncation=True, 
                          max_length=max_length, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        last_hidden = last_hidden.masked_fill(~mask.bool(), 0.0)
        emb = last_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().numpy())
    
    return np.vstack(all_embeddings)


def encode_texts_moe(texts, model, lang_idx, tokenizer, prefix, device, 
                     batch_size=16, max_length=512, num_layers=24):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [f"{prefix}: {t}" for t in batch]
        inputs = tokenizer(batch, padding=True, truncation=True, 
                          max_length=max_length, return_tensors="pt").to(device)
        
        batch_size_actual = inputs['input_ids'].shape[0]
        seq_len = inputs['input_ids'].shape[1]
        
        expert_paths = torch.zeros(batch_size_actual, seq_len, num_layers, 
                                   dtype=torch.long, device=device)
        
        for j in range(batch_size_actual):
            real_len = int(inputs['attention_mask'][j].sum().item())
            expert_paths[j, :real_len, :] = lang_idx
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                expert_paths=expert_paths
            )
        
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        last_hidden = last_hidden.masked_fill(~mask.bool(), 0.0)
        emb = last_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().numpy())
    
    return np.vstack(all_embeddings)


def compute_metrics_base(model, tokenizer, queries_data, corpus_dict, device, 
                        batch_size=16, max_length=512):
    queries = []
    all_docs = []
    qrels = []
    doc_to_idx = {}
    
    for q_idx, item in enumerate(queries_data):
        queries.append(item['query'])
        pos_doc_ids = [str(id) for id in item['positives']['doc_id']]
        for doc_id in pos_doc_ids:
            if doc_id not in doc_to_idx:
                doc_to_idx[doc_id] = len(all_docs)
                all_docs.append(corpus_dict[doc_id])
            qrels.append(Qrel(f"q_{q_idx}", f"doc_{doc_to_idx[doc_id]}", 1))
    
    q_embs = encode_texts_base(queries, model, tokenizer, "query", device, batch_size, max_length)
    d_embs = encode_texts_base(all_docs, model, tokenizer, "passage", device, batch_size, max_length)
    
    scores = np.dot(q_embs, d_embs.T)
    
    run = []
    for i in range(len(queries)):
        q_scores = scores[i]
        sorted_idx = np.argsort(q_scores)[::-1]
        for rank, doc_idx in enumerate(sorted_idx[:100]):
            run.append(ScoredDoc(f"q_{i}", f"doc_{doc_idx}", float(q_scores[doc_idx])))
    
    metrics = ir_measures.calc_aggregate([nDCG@10, MRR@10, P@10, Recall@10], qrels, run)
    return {k: float(v) for k, v in metrics.items()}


def compute_metrics_moe(model, lang_idx, tokenizer, queries_data, corpus_dict, device,
                       batch_size=16, max_length=512, num_layers=24):
    queries = []
    all_docs = []
    qrels = []
    doc_to_idx = {}
    
    for q_idx, item in enumerate(queries_data):
        queries.append(item['query'])
        pos_doc_ids = [str(id) for id in item['positives']['doc_id']]
        for doc_id in pos_doc_ids:
            if doc_id not in doc_to_idx:
                doc_to_idx[doc_id] = len(all_docs)
                all_docs.append(corpus_dict[doc_id])
            qrels.append(Qrel(f"q_{q_idx}", f"doc_{doc_to_idx[doc_id]}", 1))
    
    q_embs = encode_texts_moe(queries, model, lang_idx, tokenizer, "query", device, 
                             batch_size, max_length, num_layers)
    d_embs = encode_texts_moe(all_docs, model, lang_idx, tokenizer, "passage", device,
                             batch_size, max_length, num_layers)
    
    scores = np.dot(q_embs, d_embs.T)
    
    run = []
    for i in range(len(queries)):
        q_scores = scores[i]
        sorted_idx = np.argsort(q_scores)[::-1]
        for rank, doc_idx in enumerate(sorted_idx[:100]):
            run.append(ScoredDoc(f"q_{i}", f"doc_{doc_idx}", float(q_scores[doc_idx])))
    
    metrics = ir_measures.calc_aggregate([nDCG@10, MRR@10, P@10, Recall@10], qrels, run)
    return {k: float(v) for k, v in metrics.items()}


def load_base_model(model_name, device):
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, None


def load_moe_init(model_config_path, device):
    config = XLMRobertaMoEConfig.from_pretrained(model_config_path)
    moe_model = XLMRobertaMoEForMaskedLM.from_dense_pretrained(
        "intfloat/multilingual-e5-large", moe_num_experts=config.moe_num_experts
    )
    model = moe_model.roberta
    model.to(device)
    model.eval()
    return model, config.num_hidden_layers


def load_moe_saved(model_path, device):
    config = XLMRobertaMoEConfig.from_pretrained(model_path)
    model = XLMRobertaMoEModel(config, add_pooling_layer=False)
    model.to(device)
    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
    model.load_state_dict(state_dict)
    model.eval()
    return model, config.num_hidden_layers


def evaluate_model(model, num_layers, model_type, tokenizer, datasets_config, 
                  lang_to_idx, device, batch_size=16, max_length=512):
    results = {}
    
    for ds in datasets_config:
        logger.info(f"\nEvaluating {ds['name']}")
        queries, corpus = load_dataset(ds["path"], split="val")
        
        if model_type == "base":
            metrics = compute_metrics_base(
                model, tokenizer, queries, corpus, device, batch_size, max_length
            )
        else:
            metrics = compute_metrics_moe(
                model, ds["lang_idx"], tokenizer, queries, corpus, device,
                batch_size, max_length, num_layers
            )
        
        results[ds["name"]] = metrics
        logger.info(f"  nDCG@10: {metrics['nDCG@10']:.4f}")
    
    return results


def save_results(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model_type", type=str, default="moe_saved", 
                       choices=["base", "moe_init", "moe_saved"])
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to saved model checkpoint (for moe_saved)")
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_cfg = cfg.get("model", {})
    
    datasets_config = cfg.get("datasets", [])
    
    if not datasets_config:
        raise ValueError("No datasets specified in config file")
    
    all_langs = sorted(list(set(ds["lang"] for ds in datasets_config)))
    lang_to_idx = {lang: i for i, lang in enumerate(all_langs)}
    logger.info(f"Language mapping: {lang_to_idx}")
    
    for ds in datasets_config:
        ds["lang_idx"] = lang_to_idx[ds["lang"]]
    
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    
    if args.model_type == "base":
        model, num_layers = load_base_model("intfloat/multilingual-e5-large", device)
    elif args.model_type == "moe_init":
        model_config_path = model_cfg.get("model_config_path")
        if not model_config_path:
            raise ValueError("model_config_path not specified in config")
        model, num_layers = load_moe_init(model_config_path, device)
    elif args.model_type == "moe_saved":
        if not args.checkpoint:
            raise ValueError("checkpoint is required for moe_saved")
        model, num_layers = load_moe_saved(args.checkpoint, device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    results = evaluate_model(
        model=model,
        num_layers=num_layers,
        model_type=args.model_type.split('_')[0],
        tokenizer=tokenizer,
        datasets_config=datasets_config,
        lang_to_idx=lang_to_idx,
        device=device,
        batch_size=cfg.get("data", {}).get("batch_size", 16),
        max_length=cfg.get("data", {}).get("max_doc_length", 512)
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"results_{args.model_type}.json")
    save_results(results, output_path)


if __name__ == "__main__":
    main()