class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.average = 0


class Timer():
    """Computes elapsed time."""
    def time(self):
        return 0