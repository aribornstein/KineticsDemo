
import os
import sys
from argparse import ArgumentParser
from typing import Callable, List

import torch
from torch.utils.data.sampler import RandomSampler
import multiprocessing as mproc

import flash
from flash.core.classification import Labels
from flash.core.finetuning import NoFreeze
from flash.data.utils import download_data
from flash.utils.imports import _KORNIA_AVAILABLE, _PYTORCHVIDEO_AVAILABLE
from flash.video import VideoClassificationData, VideoClassifier

if _PYTORCHVIDEO_AVAILABLE and _KORNIA_AVAILABLE:
    import kornia.augmentation as K
    from pytorchvideo.transforms import ApplyTransformToKey, RandomShortSideScale, UniformTemporalSubsample
    from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip, Normalize
else:
    print("Please, run `pip install torchvideo kornia`")
    sys.exit(1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--backbone', type=str, default="x3d_xs")
    parser.add_argument('--download', type=bool, default=True)
    parser.add_argument('--train_folder', type=str, default=os.path.join(os.getcwd(),
                        "./data/kinetics/train"))
    parser.add_argument('--val_folder', type=str, default=os.path.join(os.getcwd(),
                        "./data/kinetics/val"))
    parser.add_argument('--predict_folder', type=str, default=os.path.join(os.getcwd(),
                        "./data/kinetics/predict"))
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--fast_dev_run', type=int, default=False)
    args = parser.parse_args()


    # 1. Download the data
    if args.download:
        # Dataset Credit:Download a video clip dataset.
        # Find more datasets at https://pytorchvideo.readthedocs.io/en/latest/data.html
        download_data("https://pl-flash-data.s3.amazonaws.com/kinetics.zip",
                      os.path.join(os.getcwd(), "data/"))


    # 2. Specify transforms to be used during training.
    # Flash helps you to place your transform exactly where you want.
    # Learn more at https://lightning-flash.readthedocs.io/en/latest/general/data.html#flash.data.process.Preprocess
    post_tensor_transform = [UniformTemporalSubsample(8), RandomShortSideScale(min_size=256, max_size=320)]
    per_batch_transform_on_device = [K.Normalize(torch.tensor([0.45, 0.45, 0.45]), torch.tensor([0.225, 0.225, 0.225]))]

    train_post_tensor_transform = post_tensor_transform + [RandomCrop(244), RandomHorizontalFlip(p=0.5)]
    val_post_tensor_transform = post_tensor_transform + [CenterCrop(244)]
    train_per_batch_transform_on_device = per_batch_transform_on_device

    def make_transform(
        post_tensor_transform: List[Callable] = post_tensor_transform,
        per_batch_transform_on_device: List[Callable] = per_batch_transform_on_device
    ):
        return {
            "post_tensor_transform": Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(post_tensor_transform),
                ),
            ]),
            "per_batch_transform_on_device": Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=K.VideoSequential(
                        *per_batch_transform_on_device, data_format="BCTHW", same_on_frame=False
                    )
                ),
            ]),
        }


    # 3. Load the data from directories.
    datamodule = VideoClassificationData.from_folders(
        train_folder=args.train_folder,
        val_folder=args.val_folder,
        predict_folder=args.predict_folder,
        train_transform=make_transform(train_post_tensor_transform),
        val_transform=make_transform(val_post_tensor_transform),
        test_transform=make_transform(val_post_tensor_transform),
        predict_transform=make_transform(val_post_tensor_transform),
        batch_size=8,
        clip_sampler="uniform",
        clip_duration=1,
        video_sampler=RandomSampler,
        decode_audio=False,
        num_workers=0,
    )

    # 4. List the available models
    print(VideoClassifier.available_backbones())
    # out: ['efficient_x3d_s', 'efficient_x3d_xs', ... ,slowfast_r50', 'x3d_m', 'x3d_s', 'x3d_xs']
    print(VideoClassifier.get_backbone_details("x3d_xs"))

    # 5. Build the VideoClassifier with a PyTorchVideo backbone.
    model = VideoClassifier(backbone=args.backbone, num_classes=datamodule.num_classes, serializer=Labels())

    # 6. Finetune the model
    trainer = flash.Trainer(max_epochs=args.max_epochs, gpus=args.gpus, fast_dev_run=args.fast_dev_run)
    trainer.finetune(model, datamodule=datamodule, strategy=NoFreeze())

    # 7. Make a prediction
    predictions = model.predict(args.predict_folder)
    print(predictions)

    # 8. Make a prediction
    trainer.save_checkpoint("video_classification.pt")

    print("Finish ! Congratulations, you used Grid Platform to train our fist VideoClassifier !!!")
