class Univariate(object):
    """ Abstract class for representing univariate distributions """

    def __init__(self):
        pass

    def fit(self):
        """ fits a univariate model and updates parameters """
        raise NotImplementedError

    def get_pdf(self, value):
        """ given a value, returns corresponding pdf value """
        raise NotImplementedError

    def get_cdf(self, value):
        """ given a value returns corresponding cdf value """
        raise NotImplementedError

    def inverse_cdf(self, value):
        """ given a cdf value, returns a value in original space """
        raise NotImplementedError

    def sample(self):
        """ returns new data point based on model """
        raise NotImplementedError

    def to_dict(self):
        """Returns parameters to replicate the distribution."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, param_dict):
        """Create new instance from dictionary."""
        raise NotImplementedError
