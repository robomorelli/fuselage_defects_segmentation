"""
Model Factory for Segmentation Architectures.
Centralized model configuration and instantiation.
"""

import torch
from mmengine import Config
from mmengine.config import ConfigDict


def freeze_backbone_layers(model, backbone_type, n_layers):
    """
    Freeze first n backbone stages.
    n_layers = 0 -> no freeze
    """
    if n_layers == 0:
        return

    if backbone_type.startswith("resnet"):
        layers = ["layer1", "layer2", "layer3", "layer4"]
        for layer_name in layers[:n_layers]:
            layer = getattr(model.backbone, layer_name)
            for param in layer.parameters():
                param.requires_grad = False

    elif backbone_type.startswith("swin"):

        # Nelle versioni recenti di MMSeg/MMEngine, SwinTransformer usa .stages invece di .layers
        for i in range(min(n_layers, len(model.backbone.stages))):
            for param in model.backbone.stages[i].parameters():
                param.requires_grad = False
    else:
        raise ValueError(f"Freeze not supported for backbone {backbone_type}")


class SegmentationModelFactory:
    """
    Factory class for building segmentation models.
    Supports: DeepLabV3+, PSPNet, U-Net, Mask2Former
    """

    SUPPORTED_ARCHITECTURES = ['deeplabv3plus', 'pspnet', 'unet', 'mask2former']

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.architecture = cfg.model.architecture
        self.backbone = cfg.model.backbone
        self.num_classes = cfg.model.num_classes
        self.crop_size = cfg.dataset.crop_size

        if self.architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Architecture {self.architecture} not supported. "
                             f"Choose from: {self.SUPPORTED_ARCHITECTURES}")

    def build(self):
        """Build and return model based on architecture."""
        from mmseg.models import build_segmentor
        from mmengine.registry import DefaultScope
        import mmseg.models
        import mmseg.datasets

        DefaultScope.get_instance('mmseg', scope_name='mmseg')

        print(f"Building {self.architecture.upper()} with backbone: {self.backbone}")

        # Build config based on architecture
        if self.architecture == 'deeplabv3plus':
            model_cfg = self._build_deeplabv3plus()
        elif self.architecture == 'pspnet':
            model_cfg = self._build_pspnet()
        elif self.architecture == 'unet':
            model_cfg = self._build_unet()
        elif self.architecture == 'mask2former':
            model_cfg = self._build_mask2former()

        # Build model
        print("Building model...")
        model = build_segmentor(model_cfg)

        freeze_layers = self.cfg.model.get("freeze_layers", 0)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        print(f"🧊 Frozen layers: {freeze_layers}")
        print(f"🔥 Trainable params: {trainable:,} / {total:,}")

        freeze_backbone_layers(
            model,
            backbone_type=self.backbone,
            n_layers=freeze_layers
        )

        print("Initializing weights...")
        model.init_weights()

        return model

    def _get_backbone_config(self):
        """Get backbone configuration based on backbone name."""
        if self.backbone == 'resnet50':
            return dict(
                type='ResNetV1c',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True
            ), 2048, 256  # in_channels, c1_channels

        elif self.backbone == 'resnet101':
            return dict(
                type='ResNetV1c',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True
            ), 2048, 256
        else:
            raise ValueError(f"Backbone {self.backbone} not supported")

    def _get_class_weights(self):
        """Get class weights from config."""
        if hasattr(self.cfg.opt, 'weights') and self.cfg.opt.weights:
            return self.cfg.opt.weights
        return None

    def _build_deeplabv3plus(self):
        """Build DeepLabV3+ configuration."""
        backbone_cfg, in_channels, c1_channels = self._get_backbone_config()
        class_weights = self._get_class_weights()

        crop_h, crop_w = (self.crop_size, self.crop_size) if isinstance(self.crop_size, int) else self.crop_size

        return ConfigDict(
            type='EncoderDecoder',
            data_preprocessor=dict(
                type='SegDataPreProcessor',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                bgr_to_rgb=False,
                pad_val=0,
                seg_pad_val=255,

            ),
            backbone=backbone_cfg,
            decode_head=dict(
                type='DepthwiseSeparableASPPHead',
                in_channels=in_channels,
                in_index=3,
                channels=512,
                dilations=(1, 12, 24, 36),
                c1_in_channels=c1_channels,
                c1_channels=48,
                dropout_ratio=0.1,
                num_classes=self.num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=class_weights
                )
            ),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=1024,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=self.num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.4,
                    class_weight=class_weights
                )
            ),
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
        )

    def _build_pspnet(self):
        """Build PSPNet configuration."""
        backbone_cfg, in_channels, _ = self._get_backbone_config()
        class_weights = self._get_class_weights()

        crop_h, crop_w = (self.crop_size, self.crop_size) if isinstance(self.crop_size, int) else self.crop_size

        return ConfigDict(
            type='EncoderDecoder',
            data_preprocessor=dict(
                type='SegDataPreProcessor',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                bgr_to_rgb=False,
                pad_val=0,
                seg_pad_val=255,

            ),
            backbone=backbone_cfg,
            decode_head=dict(
                type='PSPHead',
                in_channels=in_channels,
                in_index=3,
                channels=512,
                pool_scales=(1, 2, 3, 6),
                dropout_ratio=0.1,
                num_classes=self.num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=class_weights
                )
            ),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=1024,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=self.num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.4,
                    class_weight=class_weights
                )
            ),
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
        )

    def _build_unet(self):
        """Build U-Net configuration."""
        class_weights = self._get_class_weights()
        crop_h, crop_w = (self.crop_size, self.crop_size) if isinstance(self.crop_size, int) else self.crop_size

        return ConfigDict(
            type='EncoderDecoder',
            data_preprocessor=dict(
                type='SegDataPreProcessor',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                bgr_to_rgb=False,
                pad_val=0,
                seg_pad_val=255,

            ),
            backbone=dict(
                type='UNet',
                in_channels=3,
                base_channels=64,
                num_stages=5,
                strides=(1, 1, 1, 1, 1),
                enc_num_convs=(2, 2, 2, 2, 2),
                dec_num_convs=(2, 2, 2, 2),
                downsamples=(True, True, True, True),
                enc_dilations=(1, 1, 1, 1, 1),
                dec_dilations=(1, 1, 1, 1),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=dict(type='ReLU'),
                upsample_cfg=dict(type='InterpConv'),
                norm_eval=False
            ),
            decode_head=dict(
                type='FCNHead',
                in_channels=64,
                in_index=4,
                channels=64,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=self.num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=class_weights
                )
            ),
            train_cfg=dict(),
            test_cfg=dict(
                mode='slide',
                crop_size=(crop_h, crop_w),
                stride=(int(crop_h * 0.66), int(crop_w * 0.66))
            )
        )

    def _build_mask2former(self):
        """
        Build Mask2Former using official MMSeg configs.
        """
        print(f"✅ Building Mask2Former with backbone: {self.backbone}")

        class_weights = self._get_class_weights()
        weights_to_use = class_weights if class_weights else [1.0] * self.num_classes

        # Load base config from MMSeg based on backbone
        if self.backbone == "swin_tiny":
            base_cfg_file = "mmsegmentation/configs/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py"
        elif self.backbone == "swin_small":
            base_cfg_file = "mmsegmentation/configs/mask2former/mask2former_swin-s_8xb2-90k_cityscapes-512x1024.py"
        elif self.backbone == "resnet50":
            base_cfg_file = "mmsegmentation/configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py"
        elif self.backbone == "resnet101":
            base_cfg_file = "mmsegmentation/configs/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024.py"
        else:
            raise ValueError(f"Backbone {self.backbone} not supported for Mask2Former")

        # Load config
        print(f"Loading config from: {base_cfg_file}")
        cfg = Config.fromfile(base_cfg_file)

        # Modify for our task
        cfg.model.data_preprocessor.mean = [0.0, 0.0, 0.0]
        cfg.model.data_preprocessor.std = [1.0, 1.0, 1.0]
        cfg.model.data_preprocessor.bgr_to_rgb = False

        # Update num_classes
        cfg.model.decode_head.num_classes = self.num_classes
        if hasattr(cfg.model.decode_head, 'num_things_classes'):
            cfg.model.decode_head.num_things_classes = 0  # Semantic only
        if hasattr(cfg.model.decode_head, 'num_stuff_classes'):
            cfg.model.decode_head.num_stuff_classes = self.num_classes

        # Update class weights
        if hasattr(cfg.model.decode_head, 'loss_cls'):
            cfg.model.decode_head.loss_cls.class_weight = weights_to_use

        # Disable pretrained checkpoint loading
        cfg.load_from = None

        print(f"✓ Mask2Former config loaded with {self.num_classes} classes")

        return cfg.model

