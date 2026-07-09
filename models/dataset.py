import torch


class Dataset:
    def __init__(
        self,
    ):
        pass

    def __getitem__(self, index):
        return


class DummyDataset:
    def __iter__(self):
        for _ in range(10):
            Bs = 32
            Bt = 28

            x_s = torch.randn(Bs, 768)
            labels_s = torch.randint(
                0,
                4,
                (Bs,),
            )

            x_t = torch.randn(Bt, 768)

            yield (
                (
                    x_s,
                    labels_s,
                ),
                x_t,
            )
