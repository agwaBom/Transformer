import logging

from SCTG.models.seq2seq import Seq2seq
from SCTG.models.transformer import Transformer

logger = logging.getLogger(__name__)

class SourceCodeTextGeneration(object):

    """
    IN TRAIN.PY...

    build_word_and_char_dict
    ->  Return a dictionary from question and document words in
        provided examples.

    tgt_dict = util.build_word_and_char_dict(args,
                                        examples=train_examples + valid_examples,
                                        fields=['summary'],
                                        dict_size=args.tgt_vocab_size,
                                        no_special_token=True)
    # Init model
    model = SourceCodeTextGeneration(config.get_model_args(args), src_dict, tgt_dict)

    
    """
    def __init__(self, args, src_dict, tgt_dict, state_dict=None):
        self.args = args
        self.src_dict = src_dict
        self.args.src_vocab_size = len(src_dict)
        self.tgt_dict = tgt_dict
        self.args.tgt_vocab_size = len(tgt_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        if args.model_type == 'rnn':
            self.network = Seq2seq(self.args, tgt_dict)
        elif args.model_type == 'transformer':
            self.network = Transformer(self.args, tgt_dict)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        if state_dict:
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)
        



    # Convert a function to be static method
    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s' % filename)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        logger.info('Loading model %s' % filename)

    def init_optimizer():
        return 0

    def cuda():
        return 0

    def parallelize():
        return 0

    def updates(self, examples):
        return 0
