from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
import os
import zlib
import pickle
class HiMAPDataset(Dataset):
    def __init__(self,
                 root= '/mnt/d/av2_data/',
                 processed = 'HiMAP',
                 split='train',
                 transform=None):
        super(HiMAPDataset, self).__init__(root=root, transform=transform)
        self._num_samples = {
            'train': 199908,
            'val': 24988,
            'test': 24984,
        }[split]
        pickle_file = open(os.path.join(root, processed, '{}.dat'.format(split)), 'rb')
        self.ex_list = pickle.load(pickle_file)
        pickle_file.close()
        print(len(self.ex_list))

    def len(self):
        return self._num_samples

    def get(self, idx: int) -> HeteroData:
        data_compress = self.ex_list[idx]
        instance = pickle.loads(zlib.decompress(data_compress))
        return HeteroData(instance)

