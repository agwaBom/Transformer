import logging

logger = logging.getLogger(__name__)

class SourceCodeTextGeneration(object):
    def __init__(self, args, src_dict, tgt_dict, state_dict=None):
        self.args = args

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
