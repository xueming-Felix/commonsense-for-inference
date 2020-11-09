import argparse
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from question_answering import load_dataset, CommonsenseQAProcessor
from utils import set_seed, load_model, load_optimizer

logger = logging.getLogger(__name__)


def train(args, model, tokenizer):

    dataset = load_dataset(args, tokenizer, mode='train')
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    model, optimizer, scheduler = load_optimizer(args, model, len(dataloader))

    num_steps = 0
    best_steps = 0
    tr_loss = 0.0
    best_val_acc, best_val_loss = 0.0, 99999999999.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)

    for _ in train_iterator:

        epoch_iterator = tqdm(dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):

            model.train()

            batch = tuple(b.to(args.device) for b in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            num_steps += 1

            if args.logging_steps > 0 and num_steps % args.logging_steps == 0:
                results = evaluate(args, model, tokenizer)
                logger.info("val acc: %s, val loss: %s, at num steps: %s",
                            str(results['val_acc']),
                            str(results['val_loss']),
                            str(num_steps))
                if results["val_acc"] > best_val_acc:
                    best_val_acc, best_val_loss = results["val_acc"], results["val_loss"]
                    best_steps = num_steps

    loss = tr_loss / num_steps

    return loss, num_steps, best_steps


def evaluate(args, model, tokenizer):

    dataset = load_dataset(args, tokenizer, 'validation')
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    val_loss = 0.0
    num_steps = 0
    preds, labels = None, None

    results = {}

    for batch in tqdm(dataloader, desc="Validation"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss, logits = outputs[:2]

            val_loss += loss.mean().item()

        num_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs['labels'].detach().cpu().numpy(), axis=0)

    loss = val_loss / num_steps
    preds = np.argmax(preds, axis=1)
    acc = (preds == labels).mean()
    result = {"val_acc": acc, "val_loss": loss}
    results.update(result)

    return results


def test():
    raise NotImplemented


def main(args):
    set_seed(args.seed)

    if args.mode == 'train':
        processor = CommonsenseQAProcessor()
        num_labels = len(processor.labels)

        config_class, model_class, tokenizer_class = load_model(args.model_type)
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name,
            num_labels=num_labels, finetuning_task=args.task_name)
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name,
            do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(
            args.model_name, from_tf=bool('.ckpt' in args.model_name), config=config)

        model.to(args.device)

        train(args, model, tokenizer)

        # torch.save(model, args.save_file)
    else:
        test()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Common sense question answering")

    parser.add_argument("--mode", type=str, default='train',
                        help="Mode: <str> [ train | test ]")
    parser.add_argument("--model_type", type=str, default='bert',
                        help="Model: <str> [ bert | xlnet | roberta | gpt2 ]")
    parser.add_argument("--task_name", default=None, type=str, required=False,
                        help="The name of the task to train: <str> [ commonqa ]")
    parser.add_argument("--model_name", type=str,
                        default='/mnt/raid5/shared/bert/tensorflow/cased_L-12_H-768_A-12/bert_config.json',
                        help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--config_name", type=str,
                        default="/mnt/raid5/shared/bert/tensorflow/cased_L-12_H-768_A-12/bert_config.json",
                        help="Pre-trained config name or path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pre-trained tokenizer name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size for training.")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every n updates steps.")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--seed", type=int, default=0, help="Random seed: <int>")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    main(parser.parse_args())
