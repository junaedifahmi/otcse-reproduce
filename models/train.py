from .dataset import DummyDataset
from .ser_model import SERModel
from .trainer import OTCSETrainer

if __name__ == "__main__":
    model = SERModel()
    dataset = DummyDataset()
    trainer = OTCSETrainer(model, dataset, train_arguments)
    trainer.train()
    print("Done")
