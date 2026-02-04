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
        Build Mask2Former from scratch (no pretrained head).
        Only backbone is pretrained.
        """
        print(f"✅ Building Mask2Former from scratch with backbone: {self.backbone}")

        class_weights = self._get_class_weights()
        weights_to_use = class_weights if class_weights else [1.0] * self.num_classes

        crop_h, crop_w = (self.crop_size, self.crop_size) if isinstance(self.crop_size, int) else self.crop_size

        # Get backbone config based on backbone type
        if self.backbone == "resnet50":
            backbone_cfg = dict(
                type='ResNetV1c',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
            )
            in_channels = [256, 512, 1024, 2048]

        elif self.backbone == "resnet101":
            backbone_cfg = dict(
                type='ResNetV1c',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')
            )
            in_channels = [256, 512, 1024, 2048]

        elif self.backbone == "swin_tiny":
            backbone_cfg = dict(
                type='SwinTransformer',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                with_cp=False,

                init_cfg=dict(type='Pretrained',
                              checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth')
            )
            in_channels = [96, 192, 384, 768]

        elif self.backbone == "swin_small":
            backbone_cfg = dict(
                type='SwinTransformer',
                embed_dims=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                with_cp=False,

                init_cfg=dict(type='Pretrained',
                              checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth')
            )
            in_channels = [96, 192, 384, 768]

        elif self.backbone == "swin_base":
            backbone_cfg = dict(
                type='SwinTransformer',
                embed_dims=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(type='Pretrained',
                              checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_22kto1k_20220317-4f4dbd45.pth')
            )
            in_channels = [128, 256, 512, 1024]

        else:
            raise ValueError(f"Backbone {self.backbone} not supported for Mask2Former. "
                             f"Supported: resnet50, resnet101, swin_tiny, swin_small, swin_base")

        # Build complete config
        return ConfigDict(
            type='EncoderDecoder',
            data_preprocessor=dict(
                type='SegDataPreProcessor',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                bgr_to_rgb=False,
                pad_val=0,
                seg_pad_val=255
            ),
            backbone=backbone_cfg,
            decode_head=dict(
                type='Mask2FormerHead',
                in_channels=in_channels,
                feat_channels=256,
                out_channels=256,
                num_classes=self.num_classes,
                num_queries=100,
                num_transformer_feat_level=3,
                align_corners=False,
                pixel_decoder=dict(
                    type='MSDeformAttnPixelDecoder',
                    num_outs=3,
                    norm_cfg=dict(type='GN', num_groups=32),
                    act_cfg=dict(type='ReLU'),
                    encoder=dict(
                        type='DetrTransformerEncoder',
                        num_layers=6,
                        transformerlayers=dict(
                            type='BaseTransformerLayer',
                            attn_cfgs=dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256,
                                num_heads=8,
                                num_levels=3,
                                num_points=4,
                                im2col_step=64,
                                dropout=0.0,
                                batch_first=False,
                                norm_cfg=None,
                                init_cfg=None),
                            ffn_cfgs=dict(
                                type='FFN',
                                embed_dims=256,
                                feedforward_channels=1024,
                                num_fcs=2,
                                ffn_drop=0.0,
                                act_cfg=dict(type='ReLU', inplace=True)),
                            operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                        init_cfg=None),
                    positional_encoding=dict(
                        type='SinePositionalEncoding', num_feats=128, normalize=True),
                    init_cfg=None),
                enforce_decoder_input_project=False,
                positional_encoding=dict(
                    type='SinePositionalEncoding', num_feats=128, normalize=True),
                transformer_decoder=dict(
                    type='DetrTransformerDecoder',
                    return_intermediate=True,
                    num_layers=9,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            proj_drop=0.0,
                            dropout_layer=None,
                            batch_first=False),
                        ffn_cfgs=dict(
                            embed_dims=256,
                            feedforward_channels=2048,
                            num_fcs=2,
                            act_cfg=dict(type='ReLU', inplace=True),
                            ffn_drop=0.0,
                            dropout_layer=None,
                            add_identity=True),
                        feedforward_channels=2048,
                        operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                         'ffn', 'norm')),
                    init_cfg=None),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=2.0,
                    reduction='mean',
                    class_weight=weights_to_use),
                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='mean',
                    loss_weight=5.0),
                loss_dice=dict(
                    type='DiceLoss',
                    use_sigmoid=True,
                    activate=True,
                    reduction='mean',
                    naive_dice=False,
                    eps=1.0,
                    loss_weight=5.0)),
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
        )

