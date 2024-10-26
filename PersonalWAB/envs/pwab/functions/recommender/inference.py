import os
import time
import torch
import argparse
import numpy as np
from model import SASRec 
from utils import data_partition, build_index 
import json
from tqdm import tqdm

def predict_next_item(model, item_sequence, maxlen, itemnum, k, device='cuda', positive_indices=None):

    seq = np.zeros([maxlen], dtype=np.int32)
    seq[-len(item_sequence):] = item_sequence 

    item_indices = np.arange(1, itemnum + 1)  

    user_ids = np.array([0])  

    model.eval()  
    model.to(device)  
    with torch.no_grad():
        logits = model.predict(*[np.array(l) for l in [[user_ids], [seq], item_indices]])

        logits = -logits.cpu().numpy().flatten()  

        if positive_indices is not None:
            result = np.full(logits.shape, np.inf)
            positive_indices = [index - 1 for index in positive_indices]

            result[positive_indices] = logits[positive_indices]


        top_k_items = np.argsort(result)[:k] + 1  

    return top_k_items


if __name__ == "__main__":
    dataset_path = 'review'
    model_path = 'checkpoint/0.pth'

    u2i_index, i2u_index = build_index(dataset_path)
    dataset = data_partition(dataset_path)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('Number of users:', usernum)
    print('Number of items:', itemnum)


    maxlen = 30
    hidden_units = 50
    num_blocks = 2
    num_heads = 2
    dropout_rate = 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='review')
    parser.add_argument('--train_dir', default='review_0907')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=30, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--state_dict_path', default=None, type=str)
    args = parser.parse_args()
    model = SASRec(usernum, itemnum, args=args)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    print('model loaded from', model_path)
    model.to('cuda')

    user_history = json.load(open('../../data/', 'r'))
    s_ids = json.load(open('data/1000_ids.json', 'r'))
    product_id_map = json.load(open('data/product_id_map.json', 'r'))
    recall_1 = []
    recall_5 = []
    recall_10 = []
    total = 0
    for user_id, history in tqdm(user_history.items()):
        for item in history:
            if item['split'] == 'test':
                total += 1
                category = item['product_info']['main_category']
                timestamp = item['review']['timestamp']
                item_sequence = []
                for i in history:
                    if i['product_info']['main_category'] == category and i['review']['timestamp'] < timestamp:
                        item_sequence.append(i['product_info']['parent_asin'])
                item_sequence = [s_ids[str(i)] for i in item_sequence]
                if len(item_sequence) == 0:
                    continue
                if len(item_sequence) < 4:
                    item_sequence = [item_sequence[0]] * 3 + item_sequence
                item_sequence = item_sequence[-maxlen:]
                #positive_indices = list(product_id_map[category].values())
                positive_indices = list(s_ids.values())
                next_item = predict_next_item(model, item_sequence, maxlen, itemnum, 10, device='cuda', positive_indices=positive_indices)
                target_item = product_id_map[category][str(item['product_info']['parent_asin'])]
                if target_item in next_item[:1]:
                    recall_1.append(1)
                else:
                    recall_1.append(0)
                if target_item in next_item[:5]:
                    recall_5.append(1)
                else:
                    recall_5.append(0)
                if target_item in next_item[:10]:
                    recall_10.append(1)
                else:
                    recall_10.append(0)
    print('Total:', total)
    print('Tested:', len(recall_1))
    print('Recall@1:', sum(recall_1)/len(recall_1))
    print('Recall@5:', sum(recall_5)/len(recall_5))
    print('Recall@10:', sum(recall_10)/len(recall_10))

