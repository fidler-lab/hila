cfg = dict(
    model='mit_b4_model',
    drop_path=0.1,
    clip_grad=None,
    output_dir='checkpoints/segformer_b4',
    resume='pretrained/mit_b4.pth',
    batch_size=64,
)