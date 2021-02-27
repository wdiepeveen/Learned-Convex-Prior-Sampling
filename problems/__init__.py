class Problem:
    def __init__(self,
                 shape=None
                 ):
        if shape is None:
            raise RuntimeError("No problem size provided")
        else:
            if not isinstance(shape, tuple):
                raise RuntimeError("size is no tuple")
            else:
                self.shape = shape
