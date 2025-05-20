import argparse
from trainer import SparkConfig, Trainer
from models import RandomForest, LogisticRegressionModel, DecisionTree
from transforms import Compose, Normalize, RandomFlip

def parse_args():
    parser = argparse.ArgumentParser("Training server")
    parser.add_argument(
        "-m", "--model",
        default="rf",
        choices=["rf", "lr", "dt"],
        help="rf = RandomForest | lr = LogisticRegression | dt = DecisionTree"
    )
    return parser.parse_args()


# --------- build transform pipeline --------- #
transforms = Compose([
    RandomFlip(horizontal=True, vertical=False, p=0.35),
    Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

if __name__ == "__main__":
    args   = parse_args()
    config = SparkConfig()

    model_choice = {
        "rf": RandomForest(),
        "lr": LogisticRegressionModel(),
        "dt": DecisionTree(max_depth=None)
    }
    model = model_choice[args.model]

    trainer = Trainer(model,
                      split="train",
                      spark_conf=config,
                      transforms=transforms)
    trainer.train()
