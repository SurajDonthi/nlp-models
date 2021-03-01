# Currently use Namespace variables, Need to add a hyperparameter tuning package
from argparse import Namespace

args = Namespace()

args.description = "Training with Amazon outdoors data"
args.log_path = "./logs"
args.max_epochs = 20
args.data_path = "./data/amazon_outdoors/amazon_reviews_us_Outdoors_v1_00_rebalanced.csv.gz"
args.train_split_ratio = 0.7
# args.limit_train_batches = 0.01
# args.limit_val_batches = 0.01
# args.limit_test_batches = 0.01
args.train_batchsize = 16
args.val_batchsize = 16
args.test_batchsize = 16
args.debug = False
args.git_tag = True
