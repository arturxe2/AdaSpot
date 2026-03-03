# Global imports
import torch
import torch.nn as nn
import copy
from torchvision import transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import random
from torch.nn import functional as F

# Local imports
from model.modules import BaseRGBModel, CustomRegNetY, FCLayers, MultFCLayers, ROISelector, step
from model.shift import make_temporal_shift
from util.constants import F3SET_ELEMENTS


class AdaSpot(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()

            self._temp_arch = args.temporal_arch
            assert self._temp_arch in ['gru'], 'Only gru supported for now'

            self._feature_arch = args.feature_arch
            assert ('rny' in self._feature_arch), 'Only rny supported for now'

            # Max aggregation
            self.aggregation = args.aggregation
            assert self.aggregation in ['max'], 'Only max aggregation supported for now'

            self.lowres_loss = args.lowres_loss
            self.highres_loss = args.highres_loss
            self.roi_size = args.roi_size

            # Get main backbones (low-res and high-res) --> high-res a copy of low-res identical
            if self._feature_arch.startswith(('rny002', 'rny004', 'rny006', 'rny008')):
                backbone = CustomRegNetY(self._feature_arch)
                self.d = backbone.ds[-1]
                backbone.head.fc = nn.Identity()
            else:
                raise NotImplementedError(self._feature_arch)

            if self._feature_arch.endswith('_gsm'):
                make_temporal_shift(backbone, args.clip_len, mode='gsm', blocks_temporal = args.blocks_temporal)
            elif self._feature_arch.endswith('_gsf'):
                make_temporal_shift(backbone, args.clip_len, mode='gsf', blocks_temporal = args.blocks_temporal)

            self.lowres_backbone = backbone
            # Adapt padding convolutions backbone
            self.swap_padding(self.lowres_backbone, pad_type=args.padding)  

            self.highres_backbone = copy.deepcopy(self.lowres_backbone)   

            self.lowres_linear = nn.Sequential(
                nn.Linear(self.d, self.d // 2),
                nn.ReLU(),
                nn.Linear(self.d // 2, self.d)
            )

            self.highres_linear = nn.Sequential(
                nn.Linear(self.d, self.d // 2),
                nn.ReLU(),
                nn.Linear(self.d // 2, self.d)
            )                            

            #Positional encoding (temporal)
            self.temp_enc = nn.Parameter(torch.normal(mean = 0, std = 1 / args.clip_len, size = (args.clip_len, self.d)))

            # Temopral module & Prediction head
            if self._temp_arch == 'gru':
                self._temp_fine = nn.GRU(input_size=self.d, hidden_size=self.d, num_layers=1, batch_first=True, bidirectional=True)
                if args.dataset == 'f3set':
                    self._pred_fine = MultFCLayers(self.d * 2, F3SET_ELEMENTS)
                else:
                    self._pred_fine = FCLayers(self.d * 2, args.num_classes+1)

                # Separate heads for high-res and low-res (auxiliar supervision)
                if self.highres_loss:
                    self._temp_fine_highres = nn.GRU(input_size=self.d, hidden_size=self.d, num_layers=1, batch_first=True, bidirectional=True)
                    if args.dataset == 'f3set':
                        self._pred_fine_highres = MultFCLayers(self.d * 2, F3SET_ELEMENTS)
                    else:
                        self._pred_fine_highres = FCLayers(self.d * 2, args.num_classes+1)
                if self.lowres_loss:
                    self._temp_fine_lowres = nn.GRU(input_size=self.d, hidden_size=self.d, num_layers=1, batch_first=True, bidirectional=True)
                    if args.dataset == 'f3set':
                        self._pred_fine_lowres = MultFCLayers(self.d * 2, F3SET_ELEMENTS)
                    else:
                        self._pred_fine_lowres = FCLayers(self.d * 2, args.num_classes+1)
            
            else:
                raise NotImplementedError(self._temp_arch)
            
            #HR and LR resizing
            self.resizing_hr = T.Resize((args.hr_dim[0], args.hr_dim[1]))
            self.resizing_lr = T.Resize((args.lr_dim[0], args.lr_dim[1]))

            #HR and LR cropping (if needed)
            self.crop_hr = T.CenterCrop((args.hr_crop[0], args.hr_crop[1]))
            self.crop_lr = T.CenterCrop((args.lr_crop[0], args.lr_crop[1]))

            #Data augmentations
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
                T.RandomApply([
                    T.RandomAffine(
                        degrees=15,                    # rotate between -15 and 15 degrees
                        translate=(0.05, 0.05),        # translate up to 5% of image size
                        scale=(0.95, 1.05),            # scale between 95% and 105%
                        shear=(-5, 5, -5, 5)           # shear x between -5 and 5, y between -5 and 5
                    )
                ], p=0.25)
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

            self.unstandarization = T.Compose([
                T.Normalize(mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                    std=[1/s for s in (0.229, 0.224, 0.225)]) #Imagenet mean and std
            ])

            # RoI Selector module
            self.roi_selector = ROISelector(roi_size = args.roi_size, threshold = args.threshold, original_size = (args.hr_crop[0], args.hr_crop[1]))
            
            self.do_auxiliar_supervision = True # Set to true as default (to false when preparing model just for inference)

        def swap_padding(self, module, pad_type='zero'):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    if child.padding == (0, 0):
                        continue
                    padH, padW = child.padding

                    assert padH == padW, "Asymmetric padding not supported"

                    if pad_type == 'reflect':
                        pad_layer = nn.ReflectionPad2d(padH)
                        padding = 0
                    elif pad_type == 'replicate':
                        pad_layer = nn.ReplicationPad2d(padH)
                        padding = 0
                    else:
                        pad_layer = nn.Identity()
                        padding = padH

                    new_conv = nn.Conv2d(
                        in_channels=child.in_channels,
                        out_channels=child.out_channels,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=(child.bias is not None)
                    )

                    new_conv.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        new_conv.bias.data.copy_(child.bias.data)

                    new_layer = nn.Sequential(pad_layer, new_conv)

                    setattr(module, name, new_layer)

                else:
                    self.swap_padding(child, pad_type=pad_type)

        def forward(self, x, inference=False):
            
            x = self.normalize(x) #Normalize to 0-1

            if not inference:
                x = self.augment(x) #Augmentations per-batch

            x = self.standarize(x)
            
            # Resize and crop high-resolution 
            x_hr = self.resize(x, hr = True)
            x_hr = self.crop_hr(x_hr)

            # Resize and crop low-resolution
            x = self.resize(x)
            x = self.crop_lr(x)

            b, t, c, h, w = x.shape

            # Low-res processing
            im_feat, maps = self.lowres_backbone(x.view(-1, c, h, w), return_last_layer = True) # maps -> BT x C x H' x W'
            im_feat = im_feat.view(b, t, -1)  # (B, T, C)

            # Get high-res RoIs
            maps = maps.view(b, t, maps.shape[-3], maps.shape[-2], maps.shape[-1]) # B x T x C x H' x W'
            centers, sizes = self.roi_selector(maps.detach())
            rois = self.get_rois(x_hr, centers, sizes)

            # High-res processing of RoIs
            rois_feat = self.highres_backbone(rois.reshape(-1, c, self.roi_size[0], self.roi_size[1]))
            rois_feat = rois_feat.view(b, t, -1)
            
            # Projections
            im_feat = self.lowres_linear(im_feat)
            rois_feat = self.highres_linear(rois_feat)

            # High-res auxiliar supervision
            if self.highres_loss & self.do_auxiliar_supervision:
                im_feat_highres = rois_feat + self.temp_enc.expand(b, -1, -1)
                im_feat_highres = self._temp_fine_highres(im_feat_highres)[0]
                im_feat_highres = self._pred_fine_highres(im_feat_highres)

            # Low-res auxiliar supervision
            if self.lowres_loss & self.do_auxiliar_supervision:
                im_feat_lowres = im_feat + self.temp_enc.expand(b, -1, -1)
                im_feat_lowres = self._temp_fine_lowres(im_feat_lowres)[0]
                im_feat_lowres = self._pred_fine_lowres(im_feat_lowres)

            # Low-res + high-res fusion
            if self.aggregation == 'max':
                im_feat = torch.stack((im_feat, rois_feat), dim=-1)  # (B, T, C + C)
                im_feat = im_feat.max(dim=-1)[0]  # (B, T, C)
            else:
                raise NotImplementedError(self.aggregation)
            
            im_feat = im_feat + self.temp_enc.expand(b, -1, -1)  # Add temporal encoding

            # Temporal module and prediction head
            im_feat = self._temp_fine(im_feat)[0]
            im_feat = self._pred_fine(im_feat)

            output_dict = {}
            output_dict['im_feat'] = im_feat
            if self.highres_loss & self.do_auxiliar_supervision:
                output_dict['im_feat_highres'] = im_feat_highres
            if self.lowres_loss & self.do_auxiliar_supervision:
                output_dict['im_feat_lowres'] = im_feat_lowres

            return output_dict

        def get_rois(self, x, centers, sizes):
            b, t, c, h, w = x.shape
            ph, pw = self.roi_size

            sizes_list = self.roi_selector.sizes
            full_rois = torch.zeros((b, t, c, ph, pw), device=x.device)

            for size_h, size_w in zip(sizes_list[0], sizes_list[1]):
                mask = (sizes[..., 0] == size_h) & (sizes[..., 1] == size_w)
                if mask.sum() == 0:
                    continue
                
                # ---- Step 1: Convert normalized indicators -> integer pixel centers ----
                centers_h = (centers[..., 0] * h).long()
                centers_w = (centers[..., 1] * w).long()
                centers_h = torch.clamp(centers_h, size_h // 2, h - size_h // 2 - 1)
                centers_w = torch.clamp(centers_w, size_w // 2, w - size_w // 2 - 1)

                # ---- Step 2: Build relative offsets for roi grid ----
                dh = torch.arange(-(size_h // 2), size_h // 2, device=x.device)
                dw = torch.arange(-(size_w // 2), size_w // 2, device=x.device)

                # reshape for broadcasting
                dh = dh.view(1, 1, size_h, 1)  # (1,1,1,ph,1)
                dw = dw.view(1, 1, 1, size_w)  # (1,1,1,1,pw)

                # ---- Step 3: Absolute coordinates of each roi pixel ----    
                roi_h = centers_h[..., None, None] + dh   # (B,T,K,ph,pw)
                roi_w = centers_w[..., None, None] + dw   # (B,T,K,ph,pw)

                roi_h = roi_h.unsqueeze(2).expand(-1, -1, c, -1, w)  # (B,T,C,ph,pw)
                roi_w = roi_w.unsqueeze(2).expand(-1, -1, c, size_h, -1)

                x_exp = x.expand(b, t, c, h, w)

                rois = x_exp.gather(-2, roi_h).gather(-1, roi_w)

                if (ph != size_h) | (pw != size_w):
                    rois = F.interpolate(rois.view(-1, c, size_h, size_w), size=(ph, pw), mode='bilinear', align_corners=False)
                    rois = rois.view(b, t, c, ph, pw)
                full_rois[mask] += rois[mask]

            return full_rois

        def resize(self, x, hr = False):
            b, t, c, h, w = x.shape
            x = x.view(-1, c, h, w)  # (B, T, C, H, W) -> (B*T, C, H, W)
            if hr:
                x2 = self.resizing_hr(x)
            else:
                x2 = self.resizing_lr(x)
            return x2.view(b, t, c, x2.shape[-2], x2.shape[-1])  # (B*T, C, H, W) -> (B, T, C, H, W)
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x
        
        def unstandarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.unstandarization(x[i])
            return x

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))
            
        def clean_modules(self):
            modules = list(self._modules.keys())
            if '_temp_fine_highres' in modules:
                del self._modules['_temp_fine_highres']
            if '_pred_fine_highres' in modules:
                del self._modules['_pred_fine_highres']
            if '_pred_displ_highres' in modules:
                del self._modules['_pred_displ_highres']
            if '_temp_fine_lowres' in modules:
                del self._modules['_temp_fine_lowres']
            if '_pred_fine_lowres' in modules:
                del self._modules['_pred_fine_lowres']
            if '_pred_displ_lowres' in modules:
                del self._modules['_pred_displ_lowres']
            self.do_auxiliar_supervision = False
            print('Eliminated auxiliary supervision modules not required for inference.')


    def __init__(self, device=torch.device('cuda'), args_model=None, args_training=None, classes=None, elements=None):
        self.device = device
        args_model.lowres_loss = args_training.lowres_loss
        args_model.highres_loss = args_training.highres_loss
        self._model = AdaSpot.Impl(args=args_model)
        self._model.print_stats()
        self._dataset = args_model.dataset

        self._model.to(device)

        self._num_classes = args_model.num_classes + 1

        self._classes = classes
        self._elements = elements

        # For F3Set
        if self._elements is not None:
            self._inv_classes = {v: k for k, v in self._classes.items()}
            self._inv_elements = [{v: k for k, v in elem_dict.items()} for elem_dict in self._elements]

            self._combo_to_full_id = {}
            for event_str, class_id in self._classes.items():
                elems = event_str.split('_')
                combo = tuple(
                    self._elements[i][elems[i]] for i in range(len(elems))
                )
                self._combo_to_full_id[combo] = class_id

    def clean_modules(self):
        self._model.clean_modules()

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None,
            fg_weight=5):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        # Positive classes weights
        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs['weight'] = torch.FloatTensor(
                [1] + [fg_weight] * (self._num_classes - 1)).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device)
                if self._dataset == 'f3set':
                    labelE = batch['labelE'].float()
                    labelE = labelE.to(self.device)
                
                if 'frame2' in batch.keys():
                    frame2 = batch['frame2'].to(self.device).float()
                    label2 = batch['label2'].to(self.device)
                    if self._dataset == 'f3set':
                        labelE2 = batch['labelE2'].float()
                        labelE2 = labelE2.to(self.device)

                    l = [random.betavariate(0.2, 0.2) for _ in range(frame2.shape[0])]

                    label_dist = torch.zeros((label.shape[0], label.shape[1], self._num_classes)).to(self.device)
                    if self._dataset == 'f3set':
                        labelE_dist = [torch.zeros((labelE.shape[0], labelE.shape[2], F3SET_ELEMENTS[i-1])).to(self.device) if i > 0 else torch.zeros((labelE.shape[0], labelE.shape[2], 2)).to(self.device) for i in range(labelE.shape[1])]

                    for i in range(frame2.shape[0]):
                        frame[i] = l[i] * frame[i] + (1 - l[i]) * frame2[i]
                        lbl1 = label[i]
                        lbl2 = label2[i]

                        label_dist[i, range(label.shape[1]), lbl1] += l[i]
                        label_dist[i, range(label2.shape[1]), lbl2] += 1 - l[i]

                        if self._dataset == 'f3set':
                            for j in range(labelE.shape[1]):
                                lblE1 = labelE[i, j]
                                lblE2 = labelE2[i, j]
                                
                                if j == 0:
                                    labelE_dist[j][i, range(labelE.shape[2]), lblE1.long()] += l[i]
                                    labelE_dist[j][i, range(labelE.shape[2]), lblE2.long()] += 1 - l[i]
                                else:
                                    if (lblE1.long() != -1).any():
                                        t_idx = torch.arange(labelE.shape[2], device=lblE1.device)[lblE1.long() != -1]
                                        labelE_dist[j][i, t_idx, lblE1.long()[lblE1.long() != -1]] += l[i]
                                    if (lblE2.long() != -1).any():
                                        t_idx = torch.arange(labelE.shape[2], device=lblE2.device)[lblE2.long() != -1]
                                        labelE_dist[j][i, t_idx, lblE2.long()[lblE2.long() != -1]] += 1 - l[i]

                    label = label_dist
                    if self._dataset == 'f3set':
                        labelE = labelE_dist

                # Depends on whether mixup is used
                label = label.flatten() if len(label.shape) == 2 \
                    else label.view(-1, label.shape[-1])

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = self._model(frame, inference=inference)

                    pred = output['im_feat']
                    if self._model.highres_loss:
                        pred_highres = output['im_feat_highres']

                    if self._model.lowres_loss:
                        pred_lowres = output['im_feat_lowres']

                    loss = 0.

                    #F3Set loss - binary background/foregruand binary cross-entropy + per-category cross-entropy
                    if self._dataset == 'f3set':
                        # Loss action / background
                        pred_action = pred[0].squeeze(-1)
                        if isinstance(labelE, list):
                            label_action = labelE[0][:, :, 1]
                        else:
                            label_action = labelE[:, 0]
                        loss_final = F.binary_cross_entropy_with_logits(
                            pred_action, label_action, pos_weight=torch.tensor([fg_weight]).to(self.device)
                        )
                        # Category losses
                        for i in range(1, len(pred)):
                            
                            pred_cat = pred[i].reshape(-1, F3SET_ELEMENTS[i - 1])
                            if isinstance(labelE, list):
                                label_cat = labelE[i].reshape(-1, labelE[i].shape[-1])
                                mask = label_cat.sum(dim=1) > 0
                            else:
                                label_cat = labelE[:, i].long().reshape(-1)
                                mask = label_cat != -1
                            pred_cat = pred_cat[mask]
                            label_cat = label_cat[mask]
                            if label_cat.numel() == 0:
                                continue
                            
                            loss_cat = F.cross_entropy(
                                pred_cat, label_cat, reduction='sum') / (label_action.shape[0] * label_action.shape[1])
                            
                            loss_final += loss_cat

                    # Other dataset losses
                    else:
                        predictions = pred.reshape(-1, self._num_classes)
                        loss_final = F.cross_entropy(
                            predictions, label,
                            **ce_kwargs)
                    
                    # High-res loss
                    if self._model.highres_loss:
                        if self._dataset == 'f3set':
                            predictions_highres = pred_highres[0].squeeze(-1)
                            if isinstance(labelE, list):
                                label_action = labelE[0][:, :, 1]
                            else:
                                label_action = labelE[:, 0]
                            loss_highres = F.binary_cross_entropy_with_logits(
                                predictions_highres, label_action, pos_weight=torch.tensor([fg_weight]).to(self.device)
                            )
                            # Category losses
                            for i in range(1, len(pred_highres)):
                                pred_cat = pred_highres[i].reshape(-1, F3SET_ELEMENTS[i - 1])
                                if isinstance(labelE, list):
                                    label_cat = labelE[i].reshape(-1, labelE[i].shape[-1])
                                    mask = label_cat.sum(dim=1) > 0
                                else:
                                    label_cat = labelE[:, i].long().reshape(-1)
                                    mask = label_cat != -1
                                pred_cat = pred_cat[mask]
                                label_cat = label_cat[mask]
                                if label_cat.numel() == 0:
                                    continue
                                loss_cat = F.cross_entropy(
                                    pred_cat, label_cat, reduction='sum') / (label_action.shape[0] * label_action.shape[1])
                                loss_highres += loss_cat

                        else:
                            predictions_highres = pred_highres.reshape(-1, self._num_classes)
                            loss_highres = F.cross_entropy(
                                predictions_highres, label,
                                **ce_kwargs)
                    
                    # Low-res loss
                    if self._model.lowres_loss:
                        if self._dataset == 'f3set':
                            predictions_lowres = pred_lowres[0].squeeze(-1)
                            if isinstance(labelE, list):
                                label_action = labelE[0][:, :, 1]
                            else:
                                label_action = labelE[:, 0]
                            loss_lowres = F.binary_cross_entropy_with_logits(
                                predictions_lowres, label_action, pos_weight=torch.tensor([fg_weight]).to(self.device)
                            )
                            # Category losses
                            for i in range(1, len(pred_lowres)):
                                pred_cat = pred_lowres[i].reshape(-1, F3SET_ELEMENTS[i - 1])
                                if isinstance(labelE, list):
                                    label_cat = labelE[i].reshape(-1, labelE[i].shape[-1])
                                    mask = label_cat.sum(dim=1) > 0
                                else:
                                    label_cat = labelE[:, i].long().reshape(-1)
                                    mask = label_cat != -1
                                pred_cat = pred_cat[mask]
                                label_cat = label_cat[mask]
                                if label_cat.numel() == 0:
                                    continue
                                loss_cat = F.cross_entropy(
                                    pred_cat, label_cat, reduction='sum') / (label_action.shape[0] * label_action.shape[1])
                                loss_lowres += loss_cat
                        else:
                            predictions_lowres = pred_lowres.reshape(-1, self._num_classes)
                            loss_lowres = F.cross_entropy(
                                predictions_lowres, label,
                                **ce_kwargs)
                    
                    if self._model.highres_loss & self._model.lowres_loss:
                        loss += (loss_final + loss_highres + loss_lowres) / 3
                    
                    elif self._model.highres_loss:
                        loss += (loss_final + loss_highres) / 2

                    elif self._model.lowres_loss:
                        loss += (loss_final + loss_lowres) / 2
                    else:
                        loss += loss_final   

                if optimizer is not None:
                    step(self._model, optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq, use_amp=True):
        
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext():
                output = self._model(seq, inference=True)
            pred = output['im_feat']
            if self._dataset == 'f3set':
                pred, pred_cls = self.process_multiple_heads_prediction(
                    pred
                )
                return pred_cls.cpu().float().numpy(), pred.cpu().float().numpy()
            
            pred = torch.softmax(pred, axis=2)
            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().float().numpy(), pred.cpu().float().numpy()

    def process_multiple_heads_prediction(self, pred):
        """
        pred[0]: binary head logits        [N, T, 1]
        pred[1:]: categorical head logits  list of [N, T, Ci]

        Returns:
            output_probs: [N, T, num_full_classes]
        """

        device = pred[0].device
        N, T = pred[0].shape[:2]

        # ------------------------------------------------------------------
        # Initialize full-class log-probabilities
        # ------------------------------------------------------------------
        output_log_probs = torch.full(
            (N, T, self._num_classes),
            float("-inf"),
            device=device
        )

        # Binary head (class 0 vs rest)
        # class 0 probability = 1 - sigmoid
        binary_logits = pred[0].squeeze(-1)  # [N, T]
        log_p_action = F.logsigmoid(binary_logits)  # log P(action) = log sigmoid
        # class 0 log-probability  
        log_p_no_action = F.logsigmoid(-binary_logits)  # log P(no action) = log (1 - sigmoid)

        output_log_probs[:, :, 0] = log_p_no_action  # P(class 0)

        # Categorical heads → log-probs
        log_probs = [F.log_softmax(l, dim=-1) for l in pred[1:]]
        K = len(log_probs)

        # Valid category combinations (precomputed mapping)
        valid_combos = torch.tensor(
            list(self._combo_to_full_id.keys()),
            device=device,
            dtype=torch.long
        )  # [M, K]

        full_class_ids = torch.tensor(
            [self._combo_to_full_id[tuple(c)] for c in valid_combos.tolist()],
            device=device,
            dtype=torch.long
        )  # [M]

        M = valid_combos.shape[0]

        # Compute joint log-probabilities
        joint_log_probs = torch.zeros(N, T, M, device=device)
        for k in range(K):
            # log_probs[k]: [N, T, Ck]
            # valid_combos[:, k]: [M]
            joint_log_probs += log_probs[k][:, :, valid_combos[:, k]]

        # Multiply by P(action) (log-space addition)
        joint_log_probs += log_p_action.unsqueeze(-1)

        # Scatter joint log-probs into full class space
        output_log_probs[:, :, full_class_ids] = joint_log_probs

        # Convert to probabilities (if needed downstream)
        output_pred = torch.exp(output_log_probs)
        output_pred_cls = torch.argmax(output_pred, axis=2)

        return output_pred, output_pred_cls