cfg = dict(
    model='mit_b2_model',
    drop_path=0.1,
    clip_grad=None,
    output_dir='checkpoints/segformer_b2',
    resume='pretrained/mit_b2.pth',
    batch_size=64,
)