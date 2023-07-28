from cfg import default_config


def get_config():
    # Load the default config & then change the variables
    cfg = default_config.get_config()

    cfg.device = "cuda"
    cfg.debug = False
    cfg.deterministic = False
    cfg.num_workers = 4
    cfg.seed = 42
    cfg.hooks_enabled = False
    cfg.comment = ""
    cfg.task = "classification"
    cfg.net.type = "PointCloudResNet"

    # data dims
    cfg.dataset.params.use_normals = True
    cfg.dataset.params.use_positions = True
    cfg.net.in_channels = 6

    cfg.net.norm = "BatchNorm"
    cfg.net.block.type = "default"
    cfg.net.num_hidden = 128

    cfg.net.dropout = 0.1
    cfg.net.dropout_type = "Dropout"
    cfg.net.nonlinearity = "GELU"
    cfg.net.out_channels = 40
    cfg.net.num_blocks = 3
    cfg.net.block.width_factors = "(0,)"
    cfg.net.block.prenorm = True

    cfg.gridifier.aggregation = "mean"
    cfg.gridifier.conditioning = "rel_pos"
    cfg.gridifier.connectivity = "knn"
    cfg.gridifier.num_neighbors = 9
    cfg.gridifier.grid_resolution = 9
    cfg.gridifier.update_net.norm = "Identity"
    cfg.gridifier.update_net.type = "MLP"
    cfg.gridifier.num_backward_neighbors = 9

    cfg.gridifier.update_net.use_bias = True
    cfg.gridifier.update_net.omega_0 = 0
    cfg.gridifier.update_net.type = "MLP"
    cfg.gridifier.update_net.input_scale = 0
    cfg.gridifier.update_net.nonlinearity = "GELU"
    cfg.gridifier.update_net.num_hidden = 128
    cfg.gridifier.update_net.num_layers = 2

    cfg.gridifier.message_net.input_scale = 0
    cfg.gridifier.message_net.use_bias = True
    cfg.gridifier.message_net.type = ""
    cfg.gridifier.message_net.omega_0 = 0
    cfg.gridifier.message_net.norm = "Identity"
    cfg.gridifier.message_net.size = -1
    cfg.gridifier.message_net.type = ""
    cfg.gridifier.message_net.num_hidden = 128
    cfg.gridifier.message_net.num_layers = 2
    cfg.gridifier.message_net.nonlinearity = "GELU"

    cfg.gridifier.node_embedding.omega_0 = 0
    cfg.gridifier.node_embedding.type = "MLP"
    cfg.gridifier.node_embedding.use_bias = True
    cfg.gridifier.node_embedding.num_layers = 2
    cfg.gridifier.node_embedding.norm = "Identity"
    cfg.gridifier.node_embedding.size = -1
    cfg.gridifier.node_embedding.type = "MLP"
    cfg.gridifier.node_embedding.num_hidden = 128
    cfg.gridifier.node_embedding.input_scale = 0
    cfg.gridifier.node_embedding.nonlinearity = "GELU"

    cfg.gridifier.position_embed.norm = "Identity"
    cfg.gridifier.position_embed.size = -1
    cfg.gridifier.position_embed.type = "RFNet"
    cfg.gridifier.position_embed.nonlinearity = "GELU"
    cfg.gridifier.position_embed.use_bias = True
    cfg.gridifier.position_embed.input_scale = 0
    cfg.gridifier.position_embed.omega_0 = 0.1
    cfg.gridifier.position_embed.num_layers = 3
    cfg.gridifier.position_embed.num_hidden = 128

    cfg.conv.bias = True
    cfg.conv.type = "Conv3d"
    cfg.conv.stride = 1
    cfg.conv.kernel.bias = True
    cfg.conv.kernel.norm = "Identity"
    cfg.conv.kernel.size = 9
    cfg.conv.kernel.type = ""
    cfg.conv.kernel.omega_0 = 0
    cfg.conv.padding = "same"
    cfg.conv.kernel.chang_initialize = True
    cfg.conv.use_fft = False
    cfg.conv.kernel.size = 9
    cfg.conv.kernel.nonlinearity = "Identity"
    cfg.conv.kernel.input_scale = 0

    cfg.dataset.name = "ModelNet"
    cfg.dataset.params.val_split_as_test_split = False
    cfg.dataset.params.balance_data = False

    cfg.dataset.params.num_classes = 40
    cfg.dataset.params.num_samples = 1000

    cfg.dataset.data_dir = "data"
    cfg.dataset.augment = False

    cfg.train.do = True
    cfg.train.batch_size = 32
    cfg.train.label_smoothing = -1
    cfg.train.epochs = 60
    cfg.train.mixed_precision = False
    cfg.train.track_grad_norm = -1
    cfg.train.avail_gpus = 1
    cfg.train.grad_clip = 0
    cfg.train.accumulate_grad_steps = 1
    cfg.train.max_epochs_no_improvement = 100
    cfg.train.distributed = False

    cfg.optimizer.type ="AdamW"
    cfg.optimizer.momentum = 1
    cfg.optimizer.nesterov = False
    cfg.optimizer.weight_decay = 0
    cfg.optimizer.mask_lr_ratio = 1
    cfg.optimizer.weight_decay = 0
    cfg.optimizer.lr = 0.005

    cfg.test.before_train = False
    cfg.test.batch_size_multiplier = 1

    cfg.wandb.entity = "ck-experimental"
    cfg.wandb.project = "dense_point_clouds"

    cfg.pretrained.load = False
    cfg.pretrained.alias = "best"

    cfg.scheduler.mode = "max"
    cfg.scheduler.type = "cosine"
    cfg.scheduler.factor = 1
    cfg.scheduler.patience = -1
    cfg.scheduler.decay_steps = -1
    cfg.scheduler.warmup_epochs = 10
    cfg.scheduler.iters_per_train_epoch = 276
    cfg.scheduler.total_train_iters = 16560

    return cfg


