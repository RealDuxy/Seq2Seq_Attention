# -*- encoding: utf-8 -*-

import torch

def loss_fn(outputs, train_y):
    '''
    outputs: batch_size, seq_len, input_size
    train_y: batch_size, seq_len
    '''
    loss_obj = torch.nn.NLLLoss(reduction='mean')
    batch_loss = 0
    batch_size = train_y.shape[1]
    for i in range(batch_size):
        batch_outputs = outputs[:,i,:]
        batch_y = train_y[:,i]
        batch_loss += loss_obj(batch_outputs, batch_y)
    return batch_loss / batch_size

def lcs(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return 0
    matrix = [ [0 for x in range(len(s2))] for x in range(len(s1)) ]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                if i==0 or j==0:
                    matrix[i][j] = 1
#                     cs += s1[i]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + 1
#                     cs += s1[i]
            else:
                if i==0 or j==0:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1])

    return matrix[len(s1)-1][len(s2)-1]

def rouge(pred, target):
    beta = 100
    temp_lcs = lcs(pred, target)
    r_lcs = temp_lcs / len(target) if temp_lcs != 0 else 0
    p_lcs = temp_lcs / len(pred) if temp_lcs != 0 else 0
    up = (1 + beta**2) * r_lcs * p_lcs
    down = r_lcs + beta**2 * p_lcs + 0.001
    return up/down


if __name__ == '__main__':
    outputs = torch.randn(2, 20, 4, dtype=torch.float)
    train_y = torch.zeros(2, 20)
    print(f"---------test--------")
    print(outputs)
    print(train_y)
    print(f"loss: {loss_fn(outputs, train_y)}")