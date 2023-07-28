from cfg import default_config


def get_config():
    # Load the default config & then change the variables
    cfg = default_config.get_config()

    cfg.debug = False
    cfg.task = "segmentation"
    cfg.net.type = "PointCloudResNet"
    # dataset
    cfg.dataset.augment = False
    cfg.dataset.name = "ShapeNet"
    cfg.dataset.params.num_classes = 50
    cfg.dataset.params.use_normals = True
    cfg.dataset.params.use_positions = True
    cfg.dataset.params.val_split_as_test_split = True
    cfg.dataset.params.num_samples = 2048

    # gridifier
    cfg.gridifier.aggregation = "max"
    cfg.gridifier.conditioning = "rel_pos"
    cfg.gridifier.connectivity = "knn"
    cfg.gridifier.grid_resolution = 13
    cfg.gridifier.message_net.nonlinearity = "GELU"
    cfg.gridifier.message_net.num_hidden = 128
    cfg.gridifier.message_net.num_layers = 2
    cfg.gridifier.message_net.type = "MLP"
    cfg.gridifier.reuse_edges = True
    cfg.gridifier.same_k_forward_backward = True
    cfg.gridifier.node_embedding.nonlinearity = "GELU"
    cfg.gridifier.node_embedding.num_hidden = 128
    cfg.gridifier.node_embedding.num_layers = 2
    cfg.gridifier.node_embedding.type = "MLP"
    cfg.gridifier.num_neighbors = 9
    cfg.gridifier.position_embed.nonlinearity = "GELU"
    cfg.gridifier.position_embed.num_hidden = 128
    cfg.gridifier.position_embed.num_layers = 3
    cfg.gridifier.position_embed.omega_0 = 0.1
    cfg.gridifier.position_embed.type = "RFNet"
    cfg.gridifier.update_net.nonlinearity = "GELU"
    cfg.gridifier.update_net.num_hidden = 128
    cfg.gridifier.update_net.num_layers = 2
    cfg.gridifier.update_net.type = "MLP"

    # net
    cfg.net.dropout = 0
    cfg.net.nonlinearity = "GELU"
    cfg.net.norm = "BatchNorm"
    cfg.net.num_blocks = 4
    cfg.net.num_hidden = 128
    cfg.num_workers = 1

    cfg.conv.kernel.size = 9
    cfg.conv.type = "Conv3d"

    cfg.optimizer.lr = 0.001
    cfg.optimizer.type = "AdamW"
    cfg.optimizer.weight_decay = 0.0
    cfg.scheduler.type = "cosine"
    cfg.scheduler.warmup_epochs = 10
    cfg.seed = 42
    cfg.train.batch_size = 32
    cfg.train.epochs = 200

    return cfg



