from pyrec.utils.data_utils.data import *
import scipy.sparse
from pyrec.utils.data_utils.validation_util import generateTrainValidationRandom


class DataLoader(object):
    """docstring for DataLoader"""

    def __init__(self, train_path, test_path, data_parser,
                 ufeat_path=None, ifeat_path=None, feat_parser=None,
                 num_users=1, num_items=1,
                 transformer=None,
                 validation_generator=generateTrainValidationRandom,
                 isbinary=False):
        super(DataLoader, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.ufeat_path = ufeat_path
        self.data_parser = data_parser

        self.ifeat_path = ifeat_path
        self.feat_parser = feat_parser
        self.num_users = num_users
        self.num_items = num_items

        self.isbinary = isbinary

        self.transformer = transformer

        self.validation_generator = validation_generator

    def _loadData(self):
        d = Data()
        d.import_data(self.train_path, self.data_parser)
        d.R.data[:] = 1.0
        d.filter(self.num_users, self.num_items)
        train = d.R
        test = None
        if self.test_path is not None:
          test, _ = loadDataset(self.test_path,
                                d.users, d.items, self.data_parser)

        self.umap = d.users
        self.imap = d.items
        ufeat, ifeat = None, None
        if self.ufeat_path is not None:
            ufeat, _ = loadSideInfo(
                self.ufeat_path, self.umap, self.feat_parser)
        if self.ifeat_path is not None:
            ifeat, _ = loadSideInfo(
                self.ifeat_path, self.imap, self.feat_parser)

        return train, test, ufeat, ifeat

    def informFoldPath(self):
        self.train_path_template = self.train_path
        self.test_path_template = self.test_path

    def setFoldIndex(self, i):
        self.train_path = self.train_path_template % i
        self.test_path = self.test_path_template % i

    def _generateValidation(self, train):
        vtrain, vtest = self.validation_generator(train)
        return vtrain, vtest


class LoadIdentity(DataLoader):
    """docstring for LoadUserItem"""

    def __init__(self, train_path, test_path, data_parser,
                 ufeat_path=None, ifeat_path=None, feat_parser=None,
                 num_users=1, num_items=1,
                 transformer=None,
                 validation_generator=generateTrainValidationRandom):
        super(LoadIdentity, self).__init__(train_path, test_path, data_parser,
                                           ufeat_path, ifeat_path, feat_parser,
                                           num_users, num_items, transformer,
                                           validation_generator)

    def loadData(self):
        train, test, ufeat, ifeat = self._loadData()
        return train, train, test, ufeat, ifeat

    def generateValidation(self):
        train, test, ufeat, ifeat = self._loadData()
        vtrain, vtest = self._generateValidation(train)
        return vtrain, vtrain, vtest, ufeat, ifeat


