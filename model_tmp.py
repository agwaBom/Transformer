from numpy.lib.utils import source
import torch
import torch.optim
import torch.nn
import math

from SCTG.models.transformer import Transformer

# Copy attention
from SCTG.utils.copy_utils import collapse_copy_scores, replace_unknown, make_src_map, align

"""
Not support
1) fixed-embedding
"""

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
        """
        Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer's state dict
            use_gpu: required to move state dict to GPU        
        """
        if self.args.fix_embeddings:
            """
            models.transformer->embeddings.py

            word_lut() - word look-up table
            """
            self.network.embedder.src_word_embeddings.fix_word_lut()
            self.network.embedder.tgt_word_embeddings.fix_word_lut()
        
        if self.args.optimizer == 'sgd':
            # self.network.parameters() - Returns an iterator over module parameters.
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = torch.optim.SGD(parameters,
                                            self.args.learing_rate,
                                            momentum=self.args.momentum,
                                            weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = torch.optim.Adam(parameters,
                                            self.args.learning_rate,
                                            weight_decay=self.args.weight_decay)

        else: raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)




    ####Learning####
    def update(self, example):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        """
        Sets the module in training mode.
        mode (bool) – whether to set training mode (True) or evaluation mode (False). Default: True.
        """
        self.network.train(mode=True)

        source_map, alignment = None, None
        blank, fill = None, None

        # Collect Source map and alignment info to Enable Copy attention
        if self.args.copy_attn:
            assert 'src_map' in example and 'alignment' in example
            """
            COPY ATTENTION functions

                collapese_copy_scores
                replace_unknown
                make_src_map
                align
            """
            source_map = make_src_map(example['src_map'])
            # https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/3
            # https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
            """FAST if the asynchronous data transfer is possible"""
            source_map = source_map.cuda(non_blocking=True) if self.use_cuda else source_map

            alignment = align(example['alignment'])
            alignment = alignment.cuda(non_blocking=True) if self.use_cuda else alignment
            """
            Given scores(점수가 아니라 idx일 것 같은데..) from an expanded dictionary corresponding to a batch, 
            sums together copies, with a dictionary word when it is ambiguous.
            """
            blank, fill = collapse_copy_scores(self.tgt_dict, example['src_vocab'])


        """
        code_word_rep
        tensor([[19779, 24965, 26426, 36828, 22679],        
                [19779,    40,  9488, 29536,    40],        
                [19779, 45090, 45138,  2119, 33969],        
                [19779, 42103, 13387, 36828, 24320],        
                [19779, 11865, 36828, 37992, 19815],        
                [19779, 17615, 42653, 14998, 24729],        
                [19779, 42395, 48064, 18073, 12834],        
                [19779, 38941, 36828, 12453, 38941],        
                [19779, 31352, 18931, 38430, 26891],        
                [19779,  2024,     1, 32125, 33969],        
                [36222, 36676, 27067, 36828, 22498],        
                [19779,  2024,     1, 32125, 33969],        
                [19779,  2024,     1, 32125, 33969],        
                [19779,  2024,     1, 32125, 33969],        
                [19779, 26426, 30381, 36828, 25215],        
                [19779,  4982, 36828, 12453,  4982],        
                [19779, 26767, 48534, 36828, 48534],        
                [19779, 44728, 31629, 36828, 31629],        
                [19779,    40,  1566,   304,    40],        
                [19779, 33103, 36303, 36828, 32125],        
                [19779, 44728, 39347, 36828, 39347],        
                [19779,  9488, 32825, 36828, 36303],        
                [19779, 34898, 36828, 10925, 34898],        
                [19779,  8508, 36828, 16522, 10390],        
                [11489, 26609,     1, 48258, 33969],        
                [19779, 30541, 36828, 14215, 14196],        
                [19779, 38803, 36828, 38551, 49817],        
                [19779, 30541, 36828, 14215, 31501],        
                [19779,     1, 36828, 33672, 12287],        
                [19779, 44173, 19265, 36828,     1],        
                [19779, 33103,  7153, 40799,  8482],        
                [19779,  7153, 48788,  7153, 13679]])
        """
        code_word_rep = example['code_word_rep']
        code_char_rep = example['code_char_rep']
        code_type_rep = example['code_type_rep']
        code_mask_rep = example['code_mask_rep']
        """
        code_mask_rep
        tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        """
        code_len = example['code_len']
        """
        summ_word_rep
        tensor([[12329, 15021, 15913, 23527, 10582, 29783, 12645],        
                [12329, 27861,  1235, 23542, 17744, 29783, 12645],        
                [12329, 27070, 19459, 27070, 19224, 29783, 12645],        
                [12329, 10690, 13324, 15883,  7635, 29783, 12645],        
                [12329,  7253, 10280, 20116, 18464, 29783, 12645],        
                [12329,    21, 26507, 17744, 29318, 29783, 12645],        
                [12329, 17522,  6066, 18485,  8739, 29783, 12645],        
                [12329, 16979, 26507,  1913, 23840, 29783, 12645],        
                [12329,  8388, 25268, 29794, 21264, 29783, 12645],        
                [12329, 29271, 20461, 19459, 21358, 29783, 12645],        
                [12329, 22166, 18485, 16297, 21712, 29783, 12645],        
                [12329, 29271, 20461, 19459, 21358, 29783, 12645],        
                [12329, 29271, 20461, 19459, 21358, 29783, 12645],        
                [12329, 29271, 20461, 19459, 21358, 29783, 12645],        
                [12329, 22166, 15913, 11696, 18278, 29783, 12645],        
                [12329, 16979, 26507,  2987,  6261, 29783, 12645],        
                [12329, 16113, 13309,  3203, 29783, 12645,     0],        
                [12329, 16979, 19023, 21057, 29783, 12645,     0],        
                [12329,  8153,   922, 27396, 29783, 12645,     0],        
                [12329, 19892, 21868, 19295, 29783, 12645,     0],        
                [12329, 22166, 15027, 14416, 29783, 12645,     0],        
                [12329,  1749,  3210,  8514, 29783, 12645,     0],        
                [12329, 22166, 26507, 20993, 29783, 12645,     0],        
                [12329,  5146, 28241, 23234, 29783, 12645,     0],        
                [12329, 18485, 28953, 10690, 29783, 12645,     0],        
                [12329, 21570, 18374, 22722, 29783, 12645,     0],        
                [12329, 23110,  8153, 23569, 29783, 12645,     0],        
                [12329, 21570, 18374, 22722, 29783, 12645,     0],        
                [12329, 19526, 26097,  4107, 29783, 12645,     0],        
                [12329, 22166, 19853, 10755, 29783, 12645,     0],        
                [12329, 13256, 16530,  3203, 29783, 12645,     0],        
                [12329, 10756, 10634,  6132, 29783, 12645,     0]])
        """
        summ_word_rep = example['summ_word_rep']
        summ_char_rep = example['summ_char_rep'] # None
        """
        summ_len - summ_word_rep의 값이 들어있는 개수.
        tensor([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
        """
        summ_len = example['summ_len']
        """
        tgt_seq
        tensor([[12329, 15021, 15913, 23527, 10582, 29783, 12645],
                [12329, 27861,  1235, 23542, 17744, 29783, 12645],        
                [12329, 27070, 19459, 27070, 19224, 29783, 12645],        
                [12329, 10690, 13324, 15883,  7635, 29783, 12645],        
                [12329,  7253, 10280, 20116, 18464, 29783, 12645],        
                [12329,    21, 26507, 17744, 29318, 29783, 12645],        
                [12329, 17522,  6066, 18485,  8739, 29783, 12645],        
                [12329, 16979, 26507,  1913, 23840, 29783, 12645],        
                [12329,  8388, 25268, 29794, 21264, 29783, 12645],        
                [12329, 29271, 20461, 19459, 21358, 29783, 12645],        
                [12329, 22166, 18485, 16297, 21712, 29783, 12645],        
                [12329, 29271, 20461, 19459, 21358, 29783, 12645],        
                [12329, 29271, 20461, 19459, 21358, 29783, 12645],        
                [12329, 29271, 20461, 19459, 21358, 29783, 12645],        
                [12329, 22166, 15913, 11696, 18278, 29783, 12645],        
                [12329, 16979, 26507,  2987,  6261, 29783, 12645],        
                [12329, 16113, 13309,  3203, 29783, 12645,     0],        
                [12329, 16979, 19023, 21057, 29783, 12645,     0],        
                [12329,  8153,   922, 27396, 29783, 12645,     0],        
                [12329, 19892, 21868, 19295, 29783, 12645,     0],        
                [12329, 22166, 15027, 14416, 29783, 12645,     0],        
                [12329,  1749,  3210,  8514, 29783, 12645,     0],        
                [12329, 22166, 26507, 20993, 29783, 12645,     0],        
                [12329,  5146, 28241, 23234, 29783, 12645,     0],        
                [12329, 18485, 28953, 10690, 29783, 12645,     0],        
                [12329, 21570, 18374, 22722, 29783, 12645,     0],        
                [12329, 23110,  8153, 23569, 29783, 12645,     0],        
                [12329, 21570, 18374, 22722, 29783, 12645,     0],        
                [12329, 19526, 26097,  4107, 29783, 12645,     0],        
                [12329, 22166, 19853, 10755, 29783, 12645,     0],        
                [12329, 13256, 16530,  3203, 29783, 12645,     0],        
                [12329, 10756, 10634,  6132, 29783, 12645,     0]])
        """
        tgt_seq = example['tgt_seq']
        
        # any() - Return True if bool(x) is True for any x in the iterable.
        # example['language']에서 하나라도 None 나오면 True
        if any(l is None for l in example['language']):
            example_weights = None
        else:
            example_weights = [self.args.dataset_weights[language] for language in example['language']]
            example_weights = torch.FloatTensor(example_weights)
        
        if self.use_cuda:
            # 얘네는 항상 값이 있나보네
            code_len = code_len.cuda(non_blocking=True)
            summ_len = summ_len.cuda(non_blocking=True)
            tgt_seq = tgt_seq.cuda(non_blocking=True)
            
            if code_word_rep is not None:
                code_word_rep = code_word_rep.cuda(non_blocking=True)
            if code_char_rep is not None:
                code_char_rep = code_char_rep.cuda(non_blocking=True)
            if code_type_rep is not None:
                code_type_rep = code_type_rep.cuda(non_blocking=True)
            if code_mask_rep is not None:
                code_mask_rep = code_mask_rep.cuda(non_blocking=True)
            if summ_word_rep is not None:
                summ_word_rep = summ_word_rep.cuda(non_blocking=True)
            if summ_char_rep is not None:
                summ_char_rep = summ_char_rep.cuda(non_blocking=True)
            if example_weights is not None:
                example_weights = example_weights.cuda(non_blocking=True)                

        # training()이 True라서 바로 forward()로 가네..
        # 모든 모듈이 forward()안에서만 돌아가게 됨.
        net_loss = self.network(code_word_rep=code_word_rep,
                                code_char_rep=code_char_rep,
                                code_type_rep=code_type_rep,
                                code_len=code_len,
                                summ_word_rep=summ_word_rep,
                                summ_char_rep=summ_char_rep,
                                summ_len=summ_len,
                                tgt_seq=tgt_seq,
                                src_map=source_map,
                                alignment=alignment,
                                src_dict=self.src_dict,
                                tgt_dict=self.tgt_dict,
                                max_len=self.args.max_tgt_len,
                                blank=blank,
                                fill=fill,
                                source_vocab=example['src_vocab'],
                                code_mask_rep=code_mask_rep,
                                example_weights=example_weights)
        """
        net_loss

        {
            'loss_per_token': tensor(40.8700, devi...ackward0>), 
            'ml_loss': tensor(420.1164, dev...ackward0>)
        }
        mean()은 뭐야 대체.. 
        """
        # loss - tensor(420.1164, device='cuda:0', grad_fn=<MeanBackward0>)
        loss = net_loss['ml_loss'].mean() if self.parallel else net_loss['ml_loss']
        # loss_per_token - tensor(40.8700, device='cuda:0', grad_fn=<MeanBackward0>)
        loss_per_token = net_loss['loss_per_token'].mean() if self.parallel else net_loss['loss_per_token']

        ml_loss = loss.item() # 420.11639404296875
        loss_per_token = loss_per_token.item() # 40.87
        loss_per_token = 10 if loss_per_token > 10 else loss_per_token
        perplexity = math.exp(loss_per_token)

        # https://stackoverflow.com/questions/56799616/how-torch-tensor-backward-works
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.grad_clipping)
        # step() - Performs a single optimization step (parameter update).
        self.optimizer.step()
        # zero_grad() - Clears the gradients of all optimized
        self.optimizer.zero_grad()

        self.updates += 1

        return {
            'ml_loss': ml_loss,
            'perplexity': perplexity
        }


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