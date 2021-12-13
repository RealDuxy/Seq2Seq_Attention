# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
import os

from config import Config
from model.seq2seq import Seq2Seq
from utils.datasets import TextDataset
from utils.helper import loss_fn, rouge
from utils.prepare_data import Vocab
from tqdm import tqdm




def save_model(save_dict, save_dirs, save_limits=5):
    if not os.path.isdir(save_dirs):
        os.mkdir(save_dirs)
    ckpt_list = [x for x in os.listdir(save_dirs) if x.endswith(".pkl")]
    ckpt_list.sort()
    # 如果超出限制个数,删除最旧的model
    if save_limits <= len(ckpt_list):
        os.remove(save_dirs + f"/{ckpt_list[0]}")
    epoch = save_dict["epoch"]
    step = save_dict["step"]
    save_path = save_dirs + f"/epoch{epoch}_step{step}.pkl"
    torch.save(save_dict, save_path)
    print(f"model saved in {save_path}")


def train(model, train_dataloader, eval_dataloader, config):
    epoch_num = config.epoch_num
    vocab = model.vocab
    loss_ = loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    best_score = 0
    teacher_ratio = 0.5
    for epoch in range(epoch_num):
        teacher_ratio *= (1 - epoch / epoch_num)
        model.train()
        total_loss = 0
        print(f"-----------------epoch {epoch}-------------------")
        for step, (train_x, train_y) in tqdm(enumerate(train_dataloader)):
            outputs = model(train_x, train_y, teacher_ratio)
            loss = loss_(outputs, train_y)
            # 单向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            if step % 2 == 0:
                batch_ROUGEL = evaluate(model, eval_dataloader)
                print(f"step {step}  loss {loss.item()} average_loss {total_loss / (step + 1)} batch_ROUGEL {batch_ROUGEL}")
                if batch_ROUGEL > best_score:
                    save_dict = {'state_dict': model.state_dict(), 'epoch': epoch, 'step': step}
                    save_dir = config.out_dir + '/Seq2Seq'
                    save_model(save_dict, save_dir)



def evaluate(model, eval_dataloader):
    vocab = model.vocab
    model.eval()
    total_rouge = 0
    with torch.no_grad():
        for dev_x, target in tqdm(eval_dataloader):
            target = [x[0] for x in target]
            outputs = model(dev_x)
            predictions = outputs.argmax(2)[0]  # predictions: 64,50

            result = []
            for one_pred in predictions:
                #                 one_line_tokens = one_pred
                pred = vocab.id2word[one_pred.item()]
                if pred == "<END>" or pred == "<PAD>": break
                result.append(pred)

            total_rouge += rouge(result, target)
    return total_rouge

def predict(model, test_dataloader, save_path):
    vocab = model.vocab
    model.eval()
    all_result = []
    with torch.no_grad():
        for test_x in tqdm(test_dataloader):
            outputs = model(test_x)
            predictions = outputs.argmax(2)[0]
            result = []
            for one_pred in predictions:

                pred = vocab.id2word[one_pred.item()]
                if pred == "<END>" or pred == "<PAD>": break
                result.append(pred)
            all_result.append(result)

        with open(save_path, "w") as f:
            for line in all_result:
                f.write("".join(line)+"\n")
        print(f"predict results saved in {save_path}")


if __name__ == '__main__':
    config = Config()
    vocab = Vocab()
    vocab.reload(config.vocab_path)
    model = Seq2Seq(config, vocab)

    train_dataset = TextDataset(config, vocab, "train", 20)
    eval_dataset = TextDataset(config, vocab, "eval", 20)
    test_dataset = TextDataset(config, vocab, "test", 20)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train(model, train_dataloader, eval_dataloader, 1)
    predict(model, eval_dataloader, config.out_dir + "/Seq2Seq/test_predict.txt")

