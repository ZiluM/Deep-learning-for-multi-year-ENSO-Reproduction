import numpy as np
import os


def load_var(path, ip_len, op_len):
    ip_data_ls = []
    op_data_ls = []
    ip_data_ls1 = []
    sst = np.load(path + "/sst.npy")
    ssh = np.load(path + "/ssh.npy")
    nino34 = np.load(path + "/nino34.npy")
    print(path.split("/")[-1],"data_shape:", sst.shape, ssh.shape, nino34.shape)
    for i in range(ip_len):
        idr = -ip_len + i + 1 - op_len if -ip_len + i + 1 - op_len != 0 else None
        ip_data_sst = sst[i:idr][:, :, :, np.newaxis]
        ip_data_ssh = ssh[i:idr][:, :, :, np.newaxis]
        ip_data_ls.append(ip_data_sst)
        ip_data_ls1.append(ip_data_ssh)
    for j in range(op_len):
        idl = j + ip_len
        idr = -op_len + j + 1 if -op_len + j + 1 != 0 else None
        op_data = nino34[idl:idr][:, np.newaxis]
        op_data_ls.append(op_data)
    ip_data_ls = np.concatenate(ip_data_ls, axis=3)
    ip_data_ls1 = np.concatenate(ip_data_ls1, axis=3)
    op_data_ls = np.concatenate(op_data_ls, axis=1)
    return ip_data_ls, ip_data_ls1, op_data_ls


def load_train(path, ip_len, op_len):
    fn_ls = os.listdir(path)
    ip_data_ls_ls = []
    ip_data_ls1_ls = []
    op_data_ls_ls = []
    for fn in fn_ls:
        ip_data_ls, ip_data_ls1, op_data_ls = load_var(path + "/" + fn, ip_len, op_len)
        ip_data_ls_ls.append(ip_data_ls)
        ip_data_ls1_ls.append(ip_data_ls1)
        op_data_ls_ls.append(op_data_ls)
    ip_data_ls_ls = np.concatenate(ip_data_ls_ls, axis=0)
    ip_data_ls1_ls = np.concatenate(ip_data_ls1_ls, axis=0)
    op_data_ls_ls = np.concatenate(op_data_ls_ls, axis=0)
    print("="*80)
    print("All Data Shape:" , ip_data_ls_ls.shape,ip_data_ls1_ls.shape,op_data_ls_ls.shape)
    print("="*80)
    return ip_data_ls_ls, ip_data_ls1_ls, op_data_ls_ls

if __name__ == "__main__":
    load_train("../data/train_data",3,20)

