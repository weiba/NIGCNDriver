import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import sys
from GCN_model_ni import GraphConvolution
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def data_split(orig_adj, i, device):
    # samples_num = orig_adj.size(1)
    all_mask = torch.ones(orig_adj.size(0), orig_adj.size(1), dtype=torch.int).to(device)
    all_mask[:, i] = 0
    train_mask = all_mask
    test_mask = torch.abs(train_mask-1)
    return train_mask, test_mask

cancerType = sys.argv[1]
beta = 1500
gamma = 0.1

# cancerType = "BRCA"
# beta = 2500

exp = pd.read_table(filepath_or_buffer='./data/%s/Cancer_List/exp.txt' % cancerType, header=0, index_col=0, sep='\t')
mut = pd.read_table(filepath_or_buffer='./data/%s/Cancer_List/mut.txt' % cancerType, header=0, index_col=0, sep='\t')
mut_driver = pd.read_table(filepath_or_buffer='./data/%s/Cancer_List/mut_driver.txt' % cancerType, header=0, index_col=0, sep='\t')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

samples = mut.columns.values


exp1 = torch.tensor(exp.values.astype('float64'), dtype=torch.float).to(device)
mut1 = torch.tensor(mut.values.astype('float64'), dtype=torch.float).to(device)
mut_driver1 = torch.tensor(mut_driver.values, dtype=torch.float).to(device)


def Optimizer_process(epochs, model, optimizer, test_mask):
    for epoch in range(1, 1+epochs):
        train_out = model()
        # print(train_out)
        loss = model.loss_fun(predict=train_out)
        if epoch % 30 == 0 or epoch == epochs:
            print(f"epoch: {epoch}, loss: {loss.item()}")
        # print(f"epoch: {epoch}, loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_out = train_out.masked_select(test_mask.to(torch.bool))
    return test_out


epochs = 200
result = np.zeros((mut_driver1.size(0), mut_driver1.size(1)))
for i in range(mut_driver1.size(1)):
    print(i)
    train_mask, test_mask = data_split(mut_driver1, i, device)
    mut_driver2 = torch.mul(mut_driver1, train_mask)
    gcn_model = GraphConvolution(adj=mut_driver2,  x_feature=mut1, y_feature=exp1.T, mask=train_mask.to(torch.bool),
                                 embed_dim=128, kernel_dim=64, alpha=2, beta=beta, gamma=gamma).to(device)
    optimizer = optim.Adam(gcn_model.parameters(), lr=0.0005)
    test_result = Optimizer_process(epochs=epochs, model=gcn_model, optimizer=optimizer, test_mask=test_mask)

    print(test_result)
    result[:, i] = test_result.detach().cpu().numpy()

final_result = pd.DataFrame(result, index=mut_driver.index, columns=mut_driver.columns)
if not os.path.exists("./data/"+cancerType+"/Cancer_List/result/GCN_NI"):
    os.makedirs("./data/"+cancerType+"/Cancer_List/result/GCN_NI")
# final_result.to_csv(path_or_buf='./data/%s/Cancer_List/result/final_result.txt' % cancerType, sep='\t')
final_result.to_csv(path_or_buf='./data/%s/Cancer_List/result/GCN_NI/final_result.txt' % cancerType, sep='\t')
print(final_result)

# gamma：BRCA,HNSC,LUAD,PRAD----0.1; LUSC----0.05