from cfg import default_config


def get_config():
    # Load the default config & then change the variables
    cfg = default_config.get_config()

    # Training
    cfg.train.do = True
    cfg.train.epochs = 200
    cfg.train.batch_size = 50

    # Optimizer
    cfg.optimizer.type = "Adam"
    cfg.optimizer.lr = 1e-3

    # Scheduler
    cfg.scheduler.type = "cosine"
    cfg.scheduler.warmup_epochs = 10

    return cfg