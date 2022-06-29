_base_ = [
    '../../../_base_/datasets/ade20k_repeat.py',
    '../../../_base_/default_runtime.py',
    '../../../_base_/schedules/schedule_320k_adamw.py'
]

resume_from = "./work_dirs/hila.b2.512x512.ade.160k_S234_nohilapretraining/latest.pth"

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/segformer/mit_b2.pth',
    backbone=dict(
        type='def_b2_hila',
        info_weights=(0.5, 0.5),
        depths=[2, 2, 6, 2],
        hila_attn=[[1, 2], [2, 3], [3, 4]],
        hila_stride=[1, 1, 3, 1],
        reuse_bottomsa_pretraining=True,
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=4)

evaluation = dict(interval=161000, metric='mIoU')

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
