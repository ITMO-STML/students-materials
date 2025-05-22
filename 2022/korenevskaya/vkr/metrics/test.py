import torch 
import os
import pandas as pd
from metrics.eer import EERMetric
from tqdm import tqdm as tqdm 
from math import ceil


def prepare_pandas_protocol(protocol_path: str,
                            imposter_fname: str = "imp-enroll-test.txt",
                            targets_fname: str = "tar-enroll-test.txt",):
    names = ["enroll", "test"]
    imposters_pairs = pd.read_csv(os.path.join(protocol_path, imposter_fname), sep="\t", names=names)
    targets_pairs = pd.read_csv(os.path.join(protocol_path, targets_fname), sep="\t", names=names)
    imposters_pairs["is_target"] = 0
    targets_pairs["is_target"] = 1
    protocol = pd.concat([imposters_pairs, targets_pairs])
    return protocol


def test_network(test_loader, main_model, protocol_path, device, chunk_size=10000, min_chunk_size=2000):
    # Function to test model    
    
    protocol = prepare_pandas_protocol(protocol_path)
    main_model.eval()
    eer = EERMetric()
    print(f"Start validation on {protocol_path}...")
    with torch.no_grad():
        for data_label, data in tqdm(test_loader, total=len(test_loader.dataset)):
            steps = max(1, ceil((data.shape[-1] - min_chunk_size) \
                        / chunk_size))
            weights = torch.zeros([steps, 1])
            embs = []
            for step in range(steps):
                try:
                    cur_feat = data[..., (step *  chunk_size):((step+1) *  chunk_size)]
                    weights[step] = cur_feat.shape[-1]
                    embs.append(main_model(cur_feat.to(device)))
                except:
                    continue    
            embedding = torch.cat(embs)
            embedding = embedding * weights.to(device)
            embedding = embedding.sum(dim=0)
            eer.update(data_label[0], embedding)
    print(eer.compute(protocol))    
