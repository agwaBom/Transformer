import torch
import SCTG.inputters.constants as constants


def collapse_copy_scores(tgt_dict, src_vocabs):
    """
    Given scores from an expanded dictionary
    corresponding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = len(tgt_dict)
    blank_arr, fill_arr = [], []
    for b in range(len(src_vocabs)):
        blank = []
        fill = []
        src_vocab = src_vocabs[b]
        # Starting from 2 to ignore PAD and UNK token
        # len(src_vocab) == batch_size
        # src_vocab의 ind2tok의 0 1은 무조건 blank와 unk임
        for i in range(2, len(src_vocab)):
            sw = src_vocab[i]
            # sw = i번째 tok
            ti = tgt_dict[sw]
            # ti = sw tok에 해당하는 tgt_dict idx & tok
            # unk가 아니면 append tgt_dict에는 30000개로 이루어진 dict이므로 그 dict의 idx를 넣는다
            if ti != constants.UNK:
                blank.append(offset + i)
                fill.append(ti)
        # 그냥 크기만 같은거
        blank_arr.append(blank)
        # ti idx도 들어간거
        fill_arr.append(fill)
        """
        Fill은 이런 식으로 생김.
        00:[23812, 20982, 26256]
        01:[23812, 16119, 11389]
        02:[23812, 26754, 27746, 9222]
        03:[23812, 25075, 27746]
        04:[23812, 14109, 16256, 25295, 29012]
        05:[23812, 22721, 26429, 21935]
        ...
        """
    return blank_arr, fill_arr


def make_src_map(data):
    """ ? """
    # data 크기(길이 or batchsize)
    src_size = max([t.size(0) for t in data])
    # 각 data의 vocab의 최대 크기 tensor + 1 == 7
    src_vocab_size = max([t.max() for t in data]) + 1
    # 그만큼 공간을 비워놔
    alignment = torch.zeros(len(data), src_size, src_vocab_size)
    # sentence == 입력된 function = tensor
    # vocabulary == tokenized
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[i, j, t] = 1
    # sentence and vocab map
    return alignment


def align(data):
    """ ? """
    tgt_size = max([t.size(0) for t in data])
    alignment = torch.zeros(len(data), tgt_size).long()
    for i, sent in enumerate(data):
        alignment[i, :sent.size(0)] = sent
    return alignment


def replace_unknown(prediction, attn, src_raw):
    """ ?
        attn: tgt_len x src_len
    """
    tokens = prediction.split()
    for i in range(len(tokens)):
        if tokens[i] == constants.UNK_WORD:
            _, max_index = attn[i].max(0)
            tokens[i] = src_raw[max_index.item()]
    return ' '.join(tokens)
