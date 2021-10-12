class ConfigParameters:
    def __init__(self):
        self.mode = "train"
        self.train_continue = "off"

        self.ny = 512
        self.nx = 1024
        
        self.lr = 5e-4
        self.batch_size = 10
        self.num_epoch = 300
        self.weight_decay = 2e-4

        self.lr_decay = 0.1
        self.lr_decay_epochs = 100

        self.encoder_relu = True
        self.decoder_relu = True

        self.data_dir = "/home/mmc-server4/server/server1/Beomseong/Dataset"
        self.ckpt_dir = "./checkpoint/"
        self.log_dir = "./log"
        self.result_dir = "./result"
        