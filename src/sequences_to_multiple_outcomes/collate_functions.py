import torch
from torch.nn.utils.rnn import pad_sequence

def simple_collate_fn(batch:list)->tuple:
    item_seq, age_seq, price_seq, lengths = [], [], [], []
    for items, ages, prices in batch:
        item_seq.append(torch.tensor(items, dtype=torch.long))
        age_seq.append(torch.tensor(ages, dtype=torch.float))
        price_seq.append(torch.tensor(prices, dtype=torch.float))
        lengths.append(len(items))
    item_seq = pad_sequence(item_seq, batch_first=True, padding_value=0)
    age_seq = pad_sequence(age_seq, batch_first=True, padding_value=0.0)
    price_seq = pad_sequence(price_seq, batch_first=True, padding_value=0.0)
    max_len = item_seq.shape[1]
    # Bank note:
    # torch will broadcast torch.arange(max_len) to validate logic with torch.tensor(lengths)
    # max_len = 5
    # [[0, 1, 2, 3, 4]] < [[3], [5], [2]]
    # shape(1, 5) < shape(3, 1) | return shape(3, 5)
    mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    return item_seq, age_seq, price_seq, mask

def next_seq_collate_fn(batch:list)->tuple:
    item_seq, age_seq, price_seq, lengths = [], [], [], []
    next_item, next_age, next_price = [], [], []
    for items, ages, prices in batch:
        item_seq.append(torch.tensor(items[:-1], dtype=torch.long))
        age_seq.append(torch.tensor(ages[:-1], dtype=torch.float))
        price_seq.append(torch.tensor(prices[:-1], dtype=torch.float))
        lengths.append(len(items)-1)
        next_item.append(items[-1])
        next_age.append(ages[-1])
        next_price.append(prices[-1])

    item_seq = pad_sequence(item_seq, batch_first=True, padding_value=0)
    age_seq = pad_sequence(age_seq, batch_first=True, padding_value=0.0).unsqueeze(-1)
    price_seq = pad_sequence(price_seq, batch_first=True, padding_value=0.0).unsqueeze(-1)
    next_item = torch.tensor(next_item, dtype=torch.long)
    next_age = torch.tensor(next_age, dtype=torch.float)
    next_price = torch.tensor(next_price, dtype=torch.float)
    max_len = item_seq.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    return item_seq, age_seq, price_seq, next_item, next_age, next_price, mask