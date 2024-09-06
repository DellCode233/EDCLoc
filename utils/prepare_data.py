import numpy as np
from torch.utils.data import Dataset
from typing import Union, List, SupportsIndex, Sequence, Callable, Optional, Any, Literal
from multimethod import multimethod as singledispatchmethod


class iDataset(Dataset):
    def __init__(self, argc: Sequence[Any], y: Any = None, fe: Union[Callable, Sequence[Callable]] = None) -> None:
        super().__init__()
        self.argc = list(map(lambda x: np.array(x), argc))
        self.X = self.argc[0]
        self.y = np.array(y, dtype=np.float32) if y is not None else np.ones(shape=(len(self.X),))
        self.fe = fe if fe else lambda x: x
        assert (len(argc) <= 1) or (len(argc) == len(fe))

    @singledispatchmethod
    def __getitem__(self, index):
        print(index)
        raise NotImplementedError(f"only for int or slice or List[int]")

    @__getitem__.register
    def _(self, index: SupportsIndex):
        if len(self.argc) == 1:
            return self.fe(self.X[index]), self.y[index]
        else:
            return tuple(map(lambda x, tfe: tfe(x[index]), self.argc, self.fe)), self.y[index]

    @__getitem__.register
    def _(self, index: Union[slice, List[int]]):
        if len(self.argc) == 1:
            return (
                np.stack(list(map(self.fe, self.X[index])), axis=0),
                self.y[index],
            )
        else:
            return (tuple(map(lambda x, tfe: np.stack(list(map(tfe, x[index]))), self.argc, self.fe)), self.y[index])

    def subset(self, indices: Sequence[int]):
        return iDataset(tuple(map(lambda x: x[indices], self.argc)), y=self.y[indices], fe=self.fe)

    def __len__(self):
        return len(self.argc[0])


one_hot_matrix = np.identity(5, dtype=np.float32)
base_dict = {"A": 0, "U": 1, "G": 2, "C": 3, "T": 1, "-": 4, "N": 4}


def one_hot(seq: str):
    return one_hot_matrix[list(map(base_dict.get, seq))]  # type: ignore
