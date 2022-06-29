cfg = dict(
    model='mit_b5_model',
    drop_path=0.1,
    clip_grad=None,
    output_dir='checkpoints/segformer_b5',
    resume='pretrained/mit_b5.pth',
    batch_size=64,
)