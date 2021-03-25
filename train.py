import argparse
import numpy as np
import logging # logger
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"
import subprocess
import json
from numpy.lib.utils import source

# OrderedDict - dictionary that rememebers order
# Counter - notebook 참고.
from collections import OrderedDict, Counter

import torch
# 상태 바
from tqdm import tqdm

import SCTG.config as config
import SCTG.inputters.utils as util
import SCTG.inputters.constants as constants
import SCTG.inputters.dataset as data
import SCTG.inputters.vector as vector
from SCTG.inputters.timer import AverageMeter, Timer

from model_tmp import SourceCodeTextGeneration

from SCTG.eval.bleu.google_bleu import corpus_bleu
from SCTG.eval.rouge.rouge import Rouge
from SCTG.eval.meteor.meteor import Meteor

# init logger
logger = logging.getLogger()

def str2bool(v):
    return 0

# Just changes Number to String
# Kilo, Mega, Tera..?
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                        ['', 'K', 'M', 'B', 'T'][magnitude])


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=128,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--dataset_name', nargs='+', type=str, required=True,
                       help='Name of the experimental dataset')
    files.add_argument('--model_dir', type=str, default='/tmp/qa_models/',
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model_name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data_dir', type=str, default='/data/',
                       help='Directory of training/validation data')
    files.add_argument('--train_src', nargs='+', type=str,
                       help='Preprocessed train source file')
    files.add_argument('--train_src_tag', nargs='+', type=str,
                       help='Preprocessed train source tag file')
    files.add_argument('--train_tgt', nargs='+', type=str,
                       help='Preprocessed train target file')
    files.add_argument('--valid_src', nargs='+', type=str, required=True,
                       help='Preprocessed valid source file')
    files.add_argument('--valid_src_tag', nargs='+', type=str,
                       help='Preprocessed valid source tag file')
    files.add_argument('--valid_tgt', nargs='+', type=str, required=True,
                       help='Preprocessed valid target file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--uncase', type='bool', default=False,
                            help='Code and summary words will be lower-cased')
    preprocess.add_argument('--src_vocab_size', type=int, default=None,
                            help='Maximum allowed length for src dictionary')
    preprocess.add_argument('--tgt_vocab_size', type=int, default=None,
                            help='Maximum allowed length for tgt dictionary')
    preprocess.add_argument('--max_characters_per_token', type=int, default=30,
                            help='Maximum number of characters allowed per token')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='bleu',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display_iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')

    # Log results Learning
    log = parser.add_argument_group('Log arguments')
    log.add_argument('--print_copy_info', type='bool', default=False,
                     help='Print copy information')
    log.add_argument('--print_one_target', type='bool', default=False,
                     help='Print only one target sequence')


def set_defaults(args):
    """make sure commandline args are initialized properly"""
    # Check critical files exists
    if not args.only_test:
        # initialize
        args.train_src_files = []
        args.train_tgt_files = []
        args.train_src_tag_files = []

        """
        files.add_argument('--dataset_name', nargs='+', type=str, required=True,
                    help='Name of the experimental dataset')
        아니 파이썬 String이 들어가는데 이게 왜 Number of dataset이냐. 
        dataset을 java랑 python 두 개 다 돌릴 수 있어서~
        """
        num_dataset = len(args.dataset_name) #['python'] len == 1
        if num_dataset > 1:
            # --train_src train/code.${CODE_EXTENSION}
            if len(args.train_src) == 1:
                args.train_src = args.train_src * num_dataset
            if len(args.train_tgt) == 1:
                args.train_tgt = args.train_tgt * num_dataset
            if len(args.train_src_tag) == 1:
                args.train_src_tag = args.train_src_tag * num_dataset
        
        # one(either one of lang) or two(both)
        for i in range(num_dataset):
            # get dataset name (python || java)
            dataset_name = args.dataset_name[i]
            # data_dir : /data/python
            data_dir = os.path.join(args.data_dir, dataset_name)
            # train_src : train/code.${CODE_EXTENSION}
            train_src = os.path.join(data_dir, args.train_src[i])
            train_tgt = os.path.join(data_dir, args.train_tgt[i])
            
            # IO Exception handling
            if not os.path.isfile(train_src):
                raise IOError('No such file %s' % train_src)
            if not os.path.isfile(train_tgt):
                raise IOError('No such file %s' % train_tgt)
            """
            data.add_argument('--use_code_type', type='bool', default=False,
                              help='Use code type as additional feature for feature representations')
            """
            if args.use_code_type:
                train_src_tag = os.path.join(data_dir, args.train_src_tag[i])
                if not os.path.isfile(train_src_tag):
                    raise IOError('No such file: %s' % train_src_tag)
            else:
                train_src_tag = None
            # append source and target data path
            args.train_src_files.append(train_src)
            args.train_tgt_files.append(train_tgt)
            args.train_src_tag_files.append(train_src_tag)

            """
            files.add_argument('--dev_src_tag', nargs='+', type=str,
                                help='Preprocessed dev source tag file')
            그니까 이거 필수 아니라서 안쓰인거임... 그냥 안쓴거였음..
            """

            """
            "dev_src_files": [
                "../../data/python/dev/code.original_subtoken"
            ],
            "dev_tgt_files": [
                "../../data/python/dev/javadoc.original"
            ],
            "dev_src_tag": null,
            "dev_src_tag_files": [
                null
            ],            
            """
            args.valid_src_files = []
            args.valid_tgt_files = []
            args.valid_src_tag_files = []

            num_dataset = len(args.dataset_name)
            if num_dataset > 1:
                if len(args.valid_src) == 1:
                    args.valid_src = args.valid_src * num_dataset
                if len(args.valid_tgt) == 1:
                    args.valid_tgt = args.valid_tgt * num_dataset
                if len(args.valid_src_tag) == 1:
                    args.valid_src_tag = args.valid_src_tag * num_dataset
            
            for i in range(num_dataset):
                dataset_name = args.dataset_name[i]
                data_dir = os.path.join(args.data_dir, dataset_name)
                valid_src = os.path.join(data_dir, args.valid_src[i])
                valid_tgt = os.path.join(data_dir, args.valid_tgt[i])
                if not os.path.isfile(valid_src):
                    raise IOError('No such file %s' % valid_src)
                if not os.path.isfile(valid_tgt):
                    raise IOError('No such file %s' % valid_tgt)
                # 여기선 안쓰는거..
                if args.use_code_type:
                    valid_src_tag = os.path.join(data_dir, args.valid_src_tag[i])
                    if not os.path.isfile(valid_src_tag):
                        raise IOError('No such file: %s' % valid_src_tag)
                else:
                    valid_src_tag = None
                
                args.valid_src_files.append(valid_src)
                args.valid_tgt_files.append(valid_tgt)
                args.valid_src_tag_files.append(valid_src_tag)
    # model_dir": "../../tmp",
    # Make model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # set model name
    if not args.model_name:
        import time
        args.model_name = time.strftime("%Y%m%d-%H:%M:%S")

    suffix = '_test' if args.only_test else ''

    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.log_file = os.path.join(args.model_dir, args.model_name + suffix + '.txt')
    args.pred_file = os.path.join(args.model_dir, args.model_name + suffix + '.json')

    if args.pretrained:
        args.pretrained = os.path.join(args.model_dir, args.pretrained + '.mdl')
    """
    Make sure fix_embeddings and pretrained are consistent
    use_src_word : True
    use_tgt_word : True
    fix_embeddings : False
    pretrained : null
    """
    if args.use_src_word or args.use_tgt_word:
        if args.fix_embeddings and not args.pretrained:
            logger.warning('Warning: Embedding are random. Fix embedding is set to False')
            args.fix_embedding = False
    else:
        args.fix_embeddings = False
    
    return args

def init_from_scratch(args, train_examples, valid_examples):
    """new model, new data, new dictionary"""
    logger.info('-' * 100)
    logger.info('Build word dictionary')
    
    src_dict = util.build_word_and_char_dict(args,
                                            examples=train_examples + valid_examples,
                                            fields=['code'],
                                            dict_size=args.src_vocab_size,
                                            no_special_token=True)
    tgt_dict = util.build_word_and_char_dict(args,
                                            examples=train_examples + valid_examples,
                                            fields=['summary'],
                                            dict_size=args.tgt_vocab_size,
                                            no_special_token=True)
    logger.info('Number of words in source = %d target = %d' % (len(src_dict), len(tgt_dict)))

    # Init model
    model = SourceCodeTextGeneration(config.get_model_args(args), src_dict, tgt_dict)

    return model

def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss = AverageMeter()
    # 모델 내에서 자신의 성능을 수치화하는 내부 평가 방법
    perplexity = AverageMeter()
    epoch_time = Timer()

    # get current epoch from train loop
    current_epoch = global_stats['epoch']
    process_bar = tqdm(data_loader)

    process_bar.set_description("%s" % 'Epoch = %d [perplexity = x.xx, ml_loss = x.xx]' % current_epoch)

    # Run one epoch
    for examples in process_bar:
        batch_size = examples['batch_size']
        # If still in warmup epoch..
        if args.optimizer in ['sgd', 'adam'] and current_epoch <= args.warmup_epochs:
            current_learning_rate = global_stats['warmup_factor'] * (model.updates + 1)
            # (torch)optimizer.param_groups - a dict containing all parameter groups
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = current_learning_rate
        
        # Core.
        # model.update() - func
        # model.updates - var
        net_loss = model.update(examples)        
        ml_loss.update(net_loss['ml_loss'], batch_size)
        perplexity.update(net_loss['perplexity'], batch_size)

        # print status with tqdm
        log_info = 'Epoch = %d [perplexity = %.2f, ml_loss = %.2f]' % (current_epoch, perplexity.avg, ml_loss.avg)
        process_bar.set_description("%s" % log_info)

    logger.info('train: Epoch %d | perplexity = %.2f | ml_loss = %.2f | '
                'Time for epoch = %.2f (s)' %
                (current_epoch, perplexity.avg, ml_loss.avg, epoch_time.time()))
    
    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint', current_epoch + 1)

# train_loader & valid_loader getting in to data_loader
def validate_official(args, data_loader, model, global_stats, mode='valid'):
    eval_time = Timer()
    examples = 0
    source_list, hypothesis_list, reference_list, copy_dict = dict(), dict(), dict(), dict()

    with torch.no_grad():
        process_bar = tqdm(data_loader)
        # loop data_loader
        for i, example in enumerate(process_bar):
            batch_size = example['batch_size']
            # make space for example
            example_index = list(range(i * batch_size, (i * batch_size) + batch_size))
            # replace_unk: replace `unk` tokens while generating predictions
            # predictions is list of sentences model predicted
            predictions, targets, copy_info = model.predict(example, replace_unk=True)

            src_sequences = [code for code in example['code_text']]

            examples += batch_size

            # [model prediction], [answer], [test set] 
            for i, src, pred, tgt in zip(example_index, src_sequences, predictions, targets):
                hypothesis_list[i] = [pred]
                reference_list[i] = tgt if isinstance(tgt, list) else [tgt]
                source_list[i] = src
            
            # copy_info is coming from model.prediction() maybe i should debug what actually is it...
            if copy_info is not None:
                copy_info = copy_info.cpu().numpy().astype(int).tolist()
                for i, copy in zip(example_index, copy_info):
                    copy_dict[i] = copy

            process_bar.set_description("%s" % 'Epoch = %d [validating ... ]' % global_stats['epoch'])

    copy_dict = None if len(copy_dict) == 0 else copy_dict

    # evaluate based on [model prediction], [answer], [test set]
    # mode랑 filename은 이따 차자바

    bleu, rouge_l, meteor, precision, recall, f1 = eval_accuracies(hypothesis_list,
                                                                   reference_list,
                                                                   copy_dict,
                                                                   source_list=source_list,
                                                                   filename=args.pred_file,
                                                                   print_copy_info=args.print_copy_info,
                                                                   mode=mode)
    result = dict()
    result['bleu'] = bleu
    result['rouge_l'] = rouge_l
    result['meteor'] = meteor
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1

    if mode == 'test':
        logger.info('test valid official: '
                    'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                    (bleu, rouge_l, meteor) +
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | '
                    'examples = %d | ' %
                    (precision, recall, f1, examples) +
                    'test time = %.2f (s)' % eval_time.time())
    else:
        logger.info('validation set valid official: Epoch = %d | ' %
                    (global_stats['epoch']) +
                    'bleu = %.2f | rouge_l = %.2f | '
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | examples = %d | ' %
                    (bleu, rouge_l, precision, recall, f1, examples) +
                    'valid time = %.2f (s)' % eval_time.time())
    
    return result

def normalize_answer(s):
    """Lower text and remove extra whitespace"""
    def white_space_fix(text):
        return ' '.join(text.split())
    return white_space_fix(s.lower())

def eval_score(prediction, ground_truth):
    # True - 모델이 실제로 맞춤, False - 모델이 틀림
    # Positive - 맞다고 예측, Negative - 틀렸다고 예측

    # Precision = TP / TP + FP
    # Recall    = TP / TP + FN
    # F1        = Precision * Recall / Precision + Recall
    precision, recall, f1 = 0, 0, 0

    # if there is no ground truth
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        # common = if both token has same alphabet it gets in
        # token 단위가 단어라면 단어가 맞아야지만 들어가겠지?
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        # 총 몇 개를 맞췄냐.
        num_same = sum(common.values())

        if num_same != 0:
            # 맞다고 예측한 것들중 실제 맞은거
            precision = 1.0 * num_same / len(prediction_tokens)
            # 실제 정답이 True인 것중 모델이 얼마나 맞췄냐.
            recall = 1.0 * num_same / len(ground_truth_tokens)
            # 왜 2를 곱했는지...
            f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def compute_eval_score(prediction, ground_truth_list):
    assert isinstance(prediction, str)
    precision, recall, f1 = 0, 0, 0
    for ground_truth in ground_truth_list:
        _precision, _recall, _f1 = eval_score(prediction, ground_truth)
        # 이전 f1보다 새 f1이 더 크면 다 갈아 엎어!!!
        if _f1 > f1:
            precision, recall, f1 = _precision, _recall, _f1
    return precision, recall, f1

# 여기가 메인 evaluation
def eval_accuracies(hypothesis_list, reference_list, copy_info, source_list=None, filename=None, print_copy_info=False, mode='valid'):
    """An unofficial evalutation helper.
     Arguments:
        hypothese_list: A mapping from instance id to predicted sequences.
        reference_list: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    # same elements should be there
    assert(sorted(reference_list.keys()) == sorted(hypothesis_list.keys()))

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, ind_bleu = nltk_corpus_bleu(hypotheses, references)

    # 폴더 채로 import해서 google bleu로 계산했네
    # corpus_bleu(hypotheses, references) -> return corpus_bleu, avg_score, ind_score
    _, bleu, ind_bleu = corpus_bleu(hypothesis_list, reference_list)

    # Compute ROGUE
    rouge_l, ind_rouge = Rouge().compute_score(reference_list, hypothesis_list)

    # Compute METEOR
    if mode == 'test':
        meteor, _ = Meteor().compute_score(reference_list, hypothesis_list)
    else:
        meteor = 0

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    """
    Make JSON File like below...
    {
        "id": 0, 
        "code": "def reorder suite suite classes reverse False class count len classes suite class type suite bins [ Ordered Set for i in range class count + 1 ]partition suite by type suite classes bins reverse reverse reordered suite suite class for i in range class count + 1 reordered suite add Tests bins[i] return reordered suite", 
        "predictions": ["reorder a suite from a list of tests ."], 
        "references": ["reorders a test suite by test type ."], 
        "bleu": 0.16784459625186196, 
        "rouge_l": 0.35672514619883033
    }
    """
    json_file = open(filename, 'w') if filename else None

    # reference key를 기준으로 hypothesis list를 불러옴
    for key in reference_list.keys():
        _precision, _recall, _f1 = compute_eval_score(hypothesis_list[key][0],
                                                      reference_list[key])
        precision.update(_precision)
        recall.update(_recall)
        f1.update(_f1)

        if json_file:
            # initial setting is "print_copy_info == false" so have to try that.
            if copy_info is not None and print_copy_info:
                prediction = hypothesis_list[key][0].split()
                pred_i = [word + ' [' + str(copy_info[key][j]) + ']' for j, word in enumerate(prediction)]
                pred_i = [' '.join(pred_i)]
            else : 
                pred_i = hypothesis_list[key]
            
            log_object = OrderedDict()
            log_object['id'] = key
            if source_list is not None:
                log_object['code'] = source_list[key]
            log_object['prediction_list'] = pred_i
            log_object['reference_list'] = reference_list[key][0] if args.print_one_target else reference_list[key]
            log_object['bleu'] = ind_bleu[key]
            log_object['rouge_l'] = ind_rouge[key]

            json_file.write(json.dumps(log_object) + '\n')
    
    if json_file: json_file.close()
    return bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, recall.avg * 100, f1.avg * 100
            




    return 0

def main(args):
    #### LOAD DATA ####
    """
    03/04/2021 02:50:51 PM: [ ---------------------------------------------------------------------------------------------------- ]
    03/04/2021 02:50:51 PM: [ Load and process data files ]
    """
    logger.info('-' * 100)
    logger.info('Load and process data files')

    train_examples = []
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
            example = util.load_data(args, 
                                train_files,
                                max_examples=args.max_examples,
                                dataset_name=dataset_name)
            # mapping
            lang_name = constants.DATA_LANG_MAP[dataset_name]

            # length of loaded data
            args.dataset_weights[constants.LANG_ID_MAP[lang_name]] = len(example)

            # add data to a list ... train_examples = train_examples + example
            train_examples.extend(example)
        """
        03/04/2021 02:50:54 PM: [ Number of train examples = 55538 ]
        """
        logger.info('Number of train examples = %d' % len(train_examples))

        # Get number of train example
        args.num_train_examples = len(train_examples)

        # Dictionary = key : value pair...
        """
        03/04/2021 02:50:54 PM: [ Dataset weights = {1: 1.0} ]
        """
        for lang_id in args.dataset_weights.keys():
            weight = (1.0 * args.dataset_weights[lang_id]) / len(train_examples)
            args.dataset_weights[lang_id] = round(weight, 2)
        logger.info('Dataset weights = %s' % str(args.dataset_weights))

    valid_examples = []
    for valid_src, valid_src_tag, valid_tgt, dataset_name in \
        zip(args.valid_src_files, args.valid_src_tag_files,
            args.valid_tgt_files, args.dataset_name):
        valid_files = dict()
        valid_files['src'] = valid_src
        valid_files['src_tag'] = valid_src_tag
        valid_files['tgt'] = valid_tgt
        example = util.load_data(args,
                            valid_files,
                            max_examples=args.max_examples,
                            dataset_name=dataset_name,
                            test_split=True)
        valid_examples.extend(example)
    """
    03/04/2021 02:50:55 PM: [ Num dev examples = 18505 ]
    """
    logger.info('Number of dev example = %d' % len(valid_examples))

    #### MODEL ####
    logger.info('-' * 100)
    start_epoch = 1

    # only test
    if args.only_test:
        if args.pretrained:
            model = SourceCodeTextGeneration.load(args.pretrained)
        else:
            if not os.path.isfile(args.model_file):
                raise IOError('No such file: %s' % args.model_file)
            model = SourceCodeTextGeneration.load(args.model_file)
    else:
        # If argument is set to checkpoint and there is a model
        if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
            logger.info('Checkpoint Found')
            checkpoint_file = args.model_file + '.checkpoint'
            model, start_epoch = SourceCodeTextGeneration.load_checkpoint(checkpoint_file, args.cuda)
        else:
            # If using a pretrained model
            if args.pretrained:
                logger.info('Using pretrained model')
                model = SourceCodeTextGeneration.load(args.pretrained, args)
            else:
                logger.info('Training model from scratch')
                model = init_from_scratch(args, train_examples, valid_examples)
        
            # set optimizer
            model.init_optimizer()

            # log parameter details
            """
            03/04/2021 02:50:59 PM: [ Trainable #parameters [encoder-decoder] 44.2M [total] 86M ]
            """
            logger.info('Trainable #Parameters [encoder-decoder] {} [total] {}'.format(
                human_format(model.network.count_encoder_parameters() + 
                            model.network.count_decoder_parameters()),
                human_format(model.network.count_parameters())
            ))
            table = model.network.layer_wise_parameters()
            """
            03/04/2021 02:50:59 PM: [ Breakdown of the trainable paramters ... tables
            """
            logger.info('Breakdown of the trainable parameters\n%s' % table)

    # GPU
    if args.cuda:
        model.cuda()
    
    if args.parallel:
        model.parallelize()


    #### Data Iterator ####

    logger.info('-' * 100)
    logger.info('Make data loaders')

    # dont run if test only
    if not args.only_test:
        # PyTorch dataset class for SQuAD (and SQuAD-like) data.
        # Stanford Question Answering Dataset
        """
        train_files = dict()
        train_files['src'] = train_src
        train_files['src_tag'] = train_src_tag
        train_files['tgt'] = train_tgt
        """
        # train_files -> train_examples -> train_dataset -> train_loader
        #                train_examples -> src_dict & tgt_dict
        train_dataset = data.CommentDataset(train_examples, model)
        # "sort_by_len": true,
        if args.sort_by_len:
            # PyTorch sampler returning batched of sorted lengths (by doc and question).
            train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                    args.batch_size,
                                                    shuffle=True)
        else:
            # Random sample if sort_by_len : False
            train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
        
        """Gather a batch of individual examples into one batch."""
        # 아마 이쪽은 torch의 기능이라 documentation을 좀 봐야할듯?
        # train_dataset -> train_loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
            drop_last=args.parallel
        )

        # 왜 valid 데이터 셋에는 SequentialSampler을 넣는거지????
        valid_dataset = data.CommentDataset(valid_examples, model)
        valid_sampler = torch.utils.data.sampler.SequentialSampler(valid_dataset)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.test_batch_size,
            sampler=valid_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
            drop_last=args.parallel
        )
    
    #### Print Config ####
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' % 
                json.dumps(vars(args), indent=4, sort_keys=True))

    #### If test Only ####
    if args.only_test:
        stats = {'timer': Timer(),
                'epoch': 0,
                'best_valid': 0,
                'no_improvement': 0}
        validate_official(args, valid_loader, model, stats, mode='test')

    #### Train/Valid Loop ####
    else:
        logger.info('-' * 100)
        logger.info('Start Training')
        stats = {'timer': Timer(),
                'epoch': start_epoch,
                'best_valid': 0,
                'no_improvement': 0}
        """
        "warmup_epochs": 0,
        "warmup_steps": 0,
        """
        # warmup epoch에 따라 learning rate 설정
        if args.optimizer in ['sgd', 'adam'] and args.warmup_epochs >= start_epoch:
            logger.info("Use warmup learning rate for the %d epoch, from 0 to %s" % (args.warmup_epochs, args.learning_rate))
            num_batches = len(train_loader.dataset) // args.batch_size
            warmup_factor = (args.learning_rate + 0.) / (num_batches * args.warmup_epochs)
            stats['warmup_factor'] = warmup_factor

        # Training loop...
        for epoch in range(start_epoch, args.num_epochs + 1):
            stats['epoch'] = epoch
            if args.optimizer in ['sgd', 'adam'] and epoch > args.warmup_epochs:
                model.optimizer.param_groups[0]['lr'] = \
                    model.optimizer.param_groups[0]['lr'] * args.lr_decay
            
            train(args, train_loader, model, stats)
            result = validate_official(args, valid_loader, model, stats)

            # Save Best validation
            # 03/04/2021 02:58:31 PM: [ Best valid: bleu = 4.44 (epoch 1, 1736 updates) ]
            if result[args.valid_metric] > stats['best_valid']:
                logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                            (args.valid_metric, result[args.valid_metric],
                            stats['epoch'], model.updates))
                model.save(args.model_file)
                stats['best_valid'] = result[args.valid_metric]
                stats['no_improvement'] = 0
            else:
                """
                early_stop: 20
                20번까지 improve되지 않았다면 멈춰!!
                """
                stats['no_improvement'] += 1
                if stats['no_improvement'] >= args.early_stop:
                    break

if __name__ == '__main__':
    # For bash argument input
    parser = argparse.ArgumentParser(description='Source code to Text Generator')

    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # check if GPU is available
    args.cuda = torch.cuda.is_available()
    #args.parallel = torch.cuda.device_count() > 1
    args.parallel = False

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
    03/04/2021 02:50:51 PM: [ COMMAND: ../../main/train.py --data_workers 5 --dataset_name python --data_dir ../../data/ --model_dir ../../tmp --model_name code2jdoc --train_src train/code.original_subtoken --train_tgt train/javadoc.original --valid_src valid/code.original_subtoken --valid_tgt valid/javadoc.original --uncase True --use_src_word True --use_src_char False --use_tgt_word True --use_tgt_char False --max_src_len 400 --max_tgt_len 30 --emsize 512 --fix_embeddings False --src_vocab_size 50000 --tgt_vocab_size 30000 --share_decoder_embeddings True --max_examples -1 --batch_size 32 --test_batch_size 64 --num_epochs 200 --model_type transformer --num_head 8 --d_k 64 --d_v 64 --d_ff 2048 --src_pos_emb False --tgt_pos_emb True --max_relative_pos 32 --use_neg_dist True --nlayers 6 --trans_drop 0.2 --dropout_emb 0.2 --dropout 0.2 --copy_attn True --early_stop 20 --warmup_steps 0 --optimizer adam --learning_rate 0.0001 --lr_decay 0.99 --valid_metric bleu --checkpoint True ]
    """
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    main(args)