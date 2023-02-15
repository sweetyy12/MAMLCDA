#自定义
class FLAGSt(object):
    def __init__(self):
        self.batchsize=1
        self.datasource='miniimagenet' #from
        self.num_classes=2
        self.baseline=None
        self.pretrain_iterations=0
        self.metatrain_iterations=3209
        self.meta_batch_size=2
        self.meta_lr=0.001
        self.update_batch_size=1
        self.update_lr=0.01
        self.num_updates=5
        self.norm='batch_norm'
        self.num_filters=32
        self.conv=True
        self.max_pool=True
        self.stop_grad=False
        self.log=True
        self.logdir=r'..\logs'
        self.resume=True
        self.train=True
        self.test_iter=-1
        self.test_set=False
        self.train_update_batch_size=-1
        self.train_update_lr=-1

