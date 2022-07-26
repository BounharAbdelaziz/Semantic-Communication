import torch
device = torch.device('cpu')
class Channels():
    def AWGN(self, Tx_sig, n_var):
        Tx_sig=[1,0,1,1,1,0,1]
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        print(type(Tx_sig))
        return Rx_sig