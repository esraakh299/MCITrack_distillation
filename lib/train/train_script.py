import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
# train pipeline related
from lib.train.trainers import LTRTrainer, LTRTrainer_adapt
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.mcitrack import build_mcitrack
from lib.models.mcitrack.mcitrack import build_mcitrack_distill
from lib.train.actors import MCITrackActor, MCITrackDistillActor
from lib.utils.focal_loss import FocalLoss
# for import modules
import importlib


def run(settings):
    settings.description = 'Training script for Goku series'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg # generate cfg from lib.config
    config_module.update_config_from_file(settings.cfg_file) #update cfg from experiments
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type == "tracking":
        loader_train = build_dataloaders(cfg, settings)
    else:
        raise ValueError("illegal DATA LOADER")

    # ===== DISTILLATION TRAINING =====
    if settings.script_name == "mcitrack_distill":
        # Build teacher, student, and adaptive net
        teacher_model, student_model, adaptive_net = build_mcitrack_distill(cfg)

        # Move to GPU
        teacher_model.cuda()
        student_model.cuda()
        adaptive_net.cuda()

        # Teacher stays in eval mode
        teacher_model.eval()

        # Wrap in DDP if distributed
        if settings.local_rank != -1:
            student_model = DDP(student_model, broadcast_buffers=False,
                                device_ids=[settings.local_rank], find_unused_parameters=True)
            adaptive_net = DDP(adaptive_net, broadcast_buffers=False,
                               device_ids=[settings.local_rank], find_unused_parameters=True)
            settings.device = torch.device("cuda:%d" % settings.local_rank)
        else:
            settings.device = torch.device("cuda:0")

        # Loss functions
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT}

        # Create distillation actor
        actor = MCITrackDistillActor(
            net=student_model,
            net_teacher=teacher_model,
            adaptive_net=adaptive_net,
            objective=objective,
            loss_weight=loss_weight,
            settings=settings,
            cfg=cfg
        )

        # Dual optimizer: student + adaptive net
        optimizer, lr_scheduler, optimizer_adapt, lr_scheduler_adapt = get_optimizer_scheduler_adapt(
            student_model, adaptive_net, cfg
        )

        use_amp = getattr(cfg.TRAIN, "AMP", False)
        trainer = LTRTrainer_adapt(
            actor, [loader_train], optimizer, optimizer_adapt, settings,
            lr_scheduler=lr_scheduler, lr_scheduler_adapt=lr_scheduler_adapt, use_amp=use_amp
        )

        # Train
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)

    # ===== STANDARD TRAINING =====
    elif settings.script_name == "mcitrack":
        net = build_mcitrack(cfg)

        # wrap networks to distributed one
        net.cuda()
        if settings.local_rank != -1:
            net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=True)
            settings.device = torch.device("cuda:%d" % settings.local_rank)
        else:
            settings.device = torch.device("cuda:0")

        # Loss functions and Actors
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT}
        actor = MCITrackActor(net=net, objective=objective, loss_weight=loss_weight,
                              settings=settings, cfg=cfg)

        # Optimizer, parameters, and learning rates
        optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
        use_amp = getattr(cfg.TRAIN, "AMP", False)
        trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)

        # train process
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)

    else:
        raise ValueError("illegal script name")
