import torch
def prior_regular(network,prior_network):
    model_dict = network.state_dict()
    targt_model_dict = prior_network.state_dict()
    loss = None
    for k,v in targt_model_dict.items():
        if k in model_dict.keys():
            if loss == None:
                loss = ((v-model_dict[k])**2).reshape(-1)
            else:
                loss = torch.cat([loss, ((v-model_dict[k])**2).reshape(-1)],dim=0)
    print("prior: ",loss.sum())
    return loss.sum()
