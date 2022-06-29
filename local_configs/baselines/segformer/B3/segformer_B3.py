cfg = dict(
    model='mit_b3_model',
    drop_path=0.1,
    clip_grad=None,
    output_dir='checkpoints/segformer_b3',
    resume='pretrained/mit_b3.pth',
    batch_size=128,
)