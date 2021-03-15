import argparse
import numpy as np
import logging # logger
import sys

import torch

import SCTG.config as config
import SCTG.inputters.utils as util
import SCTG.inputters.constants as constants

# init logger
logger = logging.getLogger()



def set_defaults(args):
    return 0

def add_train_args(parser):
    return 0

def main(args):
    #### LOAD DATA ####
    """
    03/04/2021 02:50:51 PM: [ ---------------------------------------------------------------------------------------------------- ]
    03/04/2021 02:50:51 PM: [ Load and process data files ]
    """
    logger.info('-' * 100)
    logger.info('Load and process data files')

    train_exs = []
    if not args.only_test:
        args.dataset_weights = dict()
        for train_src, train_src_tag, train_tgt, dataset_name in \
            zip(args.train_src_files, args.train_src_tag_files,
            args.train_tgt_files, args.dataset_name):
            train_files = dict()
            train_files['src'] = train_src
            train_files['src_tag'] = train_src_tag
            train_files['tgt'] = train_tgt
            # load data one by one
            exs = util.load_data(args, 
                                train_files,
                                max_examples=args.max_examples,
                                dataset_name=dataset_name)
            # mapping
            lang_name = constants.DATA_LANG_MAP[dataset_name]

            # length of loaded data
            args.dataset_weights[constants.LANG_ID_MAP[lang_name]] = len(exs)

            # add data to a list ... train_exs = train_exs + exs
            train_exs.extend(exs)
        """
        03/04/2021 02:50:54 PM: [ Number of train examples = 55538 ]
        """
        logger.info('Number of train examples = %d' % len(train_exs))

        # Get number of train example
        args.num_train_examples = len(train_exs)

        # Dictionary = key : value pair...
        """
        03/04/2021 02:50:54 PM: [ Dataset weights = {1: 1.0} ]
        """
        for lang_id in args.dataset_weights.keys():
            weight = (1.0 * args.dataset_weights[lang_id]) / len(train_exs)
            args.dataset_weights[lang_id] = round(weight, 2)
        logger.info('Dataset weights = %s' % str(args.dataset_weights))

    dev_exs = []
    for dev_src, dev_src_tag, dev_tgt, dataset_name in \
        zip(args.dev_src_files, args.dev_src_tag_files,
            args.dev_tgt_files, args.dataset_name):
        dev_files = dict()
        dev_files['src'] = dev_src
        dev_files['src_tag'] = dev_src_tag
        dev_files['tgt'] = dev_tgt
        exs = util.load_data(args,
                            dev_files,
                            max_examples=args.max_examples,
                            dataset_name=dataset_name,
                            test_split=True)
        dev_exs.extend(exs)
    """
    03/04/2021 02:50:55 PM: [ Num dev examples = 18505 ]
    """
    logger.info('Number of dev example = %d' % len(dev_exs))

    #### MODEL ####

if __name__ == 'main':
    # For bash argument input
    parser = argparse.ArgumentParser(description='Source code to Text Generator')

    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # check if GPU is available
    args.cuda = torch.cuda.is_available()
    args.parallel = torch.cude.device_count() > 1

    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # If GPU
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logger level to INFO
    logger.setLevel(logging.INFO)

    format = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')

    console = logging.StreamHandler()
    console.setFormatter(format)

    logger.addHandler(console)

    # if logfile is made(should be made)
    # save log to args.log_file
    if args.log_file:
        if args.checkpoint:
            # If mode is not specified, 'a'
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            # write..?
            logfile = logging.FileHandler(args.log_file, 'w')
        logger.addHandler(logfile)
    """
    03/04/2021 02:50:51 PM: [ COMMAND: ../../main/train.py --data_workers 5 --dataset_name python --data_dir ../../data/ --model_dir ../../tmp --model_name code2jdoc --train_src train/code.original_subtoken --train_tgt train/javadoc.original --dev_src dev/code.original_subtoken --dev_tgt dev/javadoc.original --uncase True --use_src_word True --use_src_char False --use_tgt_word True --use_tgt_char False --max_src_len 400 --max_tgt_len 30 --emsize 512 --fix_embeddings False --src_vocab_size 50000 --tgt_vocab_size 30000 --share_decoder_embeddings True --max_examples -1 --batch_size 32 --test_batch_size 64 --num_epochs 200 --model_type transformer --num_head 8 --d_k 64 --d_v 64 --d_ff 2048 --src_pos_emb False --tgt_pos_emb True --max_relative_pos 32 --use_neg_dist True --nlayers 6 --trans_drop 0.2 --dropout_emb 0.2 --dropout 0.2 --copy_attn True --early_stop 20 --warmup_steps 0 --optimizer adam --learning_rate 0.0001 --lr_decay 0.99 --valid_metric bleu --checkpoint True ]
    """
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    main(args)

