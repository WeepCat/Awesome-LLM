import os
import argparse
import time
from datasets import load_dataset
from RLHF.reward_modeling import DefaultRewardDataCollator, accuracy, BradleyTerryRMTrainer
from utils.util import get_logger
from transformers import TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from config import trainer_configs
# env
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['HF_ENDPOINT'] = '/root/autodl-tmp/cache/'
os.environ['access_token'] = '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'

parser = argparse.ArgumentParser(description="")
parser.add_argument('--model_name', type=str, required=True, help="Pre-trained model name")
parser.add_argument('--train_dataset_name', type=str, required=True, help="Dataset name (e.g., 'Anthropic/hh-rlhf')")
parser.add_argument('--eval_dataset_name', type=str, required=True, help="Dataset name (e.g., 'Anthropic/hh-rlhf')")
parser.add_argument('--per_device_train_batch_size', type=int, default=4, help="Batch size for training")
parser.add_argument('--per_device_eval_batch_size', type=int, default=64, help="Batch size for evaluation")
parser.add_argument('--cache_dir', type=str, default='/root/autodl-tmp/cache/', help="")
parser.add_argument('--logging_dir', type=str, default='./logs/train/', help="")
parser.add_argument('--output_dir', type=str, default='./results/train/', help="Directory to save evaluation results")
parser.add_argument('--save_path', type=str, default="./models/", help="Save path")
parser.add_argument('--shutdown', type=bool, default=False, help="Shutdown after training")
args = parser.parse_args()
logging_dir = str(os.path.join(args.logging_dir, args.model_name))
output_dir = str(os.path.join(args.output_dir, args.model_name))


if __name__ == "__main__":
    logger = get_logger(logging_dir)

    logger.info(f"Loading model {args.model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                               cache_dir=args.cache_dir,
                                                               use_flash_attention_2=args.use_flash_attention_2,
                                                               token=os.getenv('access_token'))
    logger.info(f"Loading tokenizer {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              cache_dir=args.cache_dir,
                                              token=os.getenv('access_token')
                                              )

    train_ds = load_dataset(args.train_dataset_name, split="train")
    test_ds = load_dataset(args.eval_dataset_name, split="test")

    logger.info("Training set: %d, Test set: %d" % (len(train_ds), len(test_ds)))
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        **trainer_configs
    )

    model.config.use_cache = not trainer_configs["gradient_checkpointing"]
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    trainer = BradleyTerryRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=accuracy,
        data_collator=DefaultRewardDataCollator(
            tokenizer=tokenizer),
    )

    trainer.train()

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    save_path = os.path.join(args.save_path, f"{args.model_name}", now)
    logger.info(f"Saving last checkpoint of the model to {save_path}...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

    if args.shutdown:
        os.system("/usr/bin/shutdown")
