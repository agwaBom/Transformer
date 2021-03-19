import torch

from SCTG.models.transformer import Transformer



class SourceCodeTextGeneration(object):
    """
    DICT EXAMPLE

    _word_char_ids:array([[256,  60,  98, ..., 258, 258, 258],
                        [256,  60, 117, ..., 258, 258, 258],
                        [256,  98,  97, ..., 258, 258, 258],
                        ...,
                        [256, 101, 109, ..., 258, 258, 258],
                        [256, 116, 101, ..., 258, 258, 258],
                        [256, 105, 110, ..., 258, 258, 258]], dtype=int32)
    _max_word_length:30
    _convert_word_to_char_ids:<bound method UnicodeCharsVocabulary._convert_word_to_char_ids of <SCTG.inputters.vocabulary.UnicodeCharsVocabulary object at 0x7fd2c0c40ca0>>
    word_char_ids:array([[256,  60,  98, ..., 258, 258, 258],
                        [256,  60, 117, ..., 258, 258, 258],
                        [256,  98,  97, ..., 258, 258, 258],
                        ...,
                        [256, 101, 109, ..., 258, 258, 258],
                        [256, 116, 101, ..., 258, 258, 258],
                        [256, 105, 110, ..., 258, 258, 258]], dtype=int32)
    tok2ind:{'"': 31252, '""': 46037, '"#': 27104, '"#f': 29687, '"$': 47444, '"%': 48035, '"%rcan\'thavedocstrings"': 24994, '"%s': 22821, '"%s"': 17760, '"%s"\'': 38481, '"%s"/>\'': 46276, '"%s">': 30556, '"%s">%s&nbsp': 20048, '"%s">%s</a>\'': 15244, ...}
    pad_char:258
    max_word_length:30
    ind2tok:{0: '<blank>', 1: '<unk>', 2: 'bary', 3: "'>'", 4: 'successful', 5: 'othermp', 6: 'nbrdict', 7: 'tmplstr', 8: 'item[', 9: '[test]', 10: 'iterable[', 11: 'provider', 12: 'conlltags', 13: "info['shell']", ...}
    eow_char:257
    bow_char:256
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

        if args.model_type == 'transformer':
            self.network = Transformer(self.args, tgt_dict)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        # Load saved state 아직 안 써본 듯.
        if state_dict:
            # Load buffer separately(???)
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                """
                Copies parameters and buffers from state_dict into this module and its descendants. 
                If strict is True, then the keys of state_dict must exactly match the keys returned by this module's ~torch.nn.Module.state_dict function.
                
                Param:
                state_dict (dict): a dict containing parameters and persistent buffers.
                strict (bool, optional): whether to strictly enforce that the keys in state_dict match the keys returned by this module's
                                        ~torch.nn.Module.state_dict function. Default: True
                """
                self.network.load_state_dict(state_dict)

                """                
                Adds a buffer to the module.
                This is typically used to register a buffer that should not to be considered a model parameter.
                
                Args:
                name (string): name of the buffer. The buffer can be accessed from this module using the given name
                tensor (Tensor): buffer to be registered.
                persistent (bool): whether the buffer is part of this module's state_dict.
                """
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        return 0

    ####Learning####
    def update(self, example):
        return 0

    ####Prediction####
    def predict(self, example, replace_unk=False):
        return 0

    ####Save&Loading####
    def save(self, filename):
        return 0

    def checkpoint(self, filename, epoch):
        return 0
    
    @staticmethod
    def load(filename, new_args=None):
        return 0

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        return 0
    
    ####Runtime####
    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """use data parallel to copy the model across several GPUs"""
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)    