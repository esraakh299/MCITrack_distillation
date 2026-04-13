from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_iou
import torch
import torch.nn.functional as F
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.gs import gumbel_softmax


class MCITrackDistillActor(BaseActor):
    """Actor for training MCITrack with Target-aware Adaptive Distillation (TAD).

    Uses a frozen teacher model and an adaptive network to selectively transfer
    knowledge to the student model based on per-sample difficulty.
    """

    def __init__(self, net, net_teacher, adaptive_net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.net_teacher = net_teacher
        self.adaptive_net = adaptive_net
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
        returns:
            loss        - the student training loss (standard + KD)
            loss_adapt  - the adaptive network loss
            status      - dict containing detailed losses
        """
        # forward pass (teacher + student + adaptive)
        out_dict_student, out_dict_teacher, adapt_logits, adapt_decisions = self.forward_pass(data)

        # compute standard losses on student
        loss_standard, status = self.compute_losses(out_dict_student, data)

        # compute distillation losses (KD + feature)
        loss_kd, loss_feat, loss_adapt, distill_status = self.compute_losses_distill(
            out_dict_student, out_dict_teacher, adapt_logits, adapt_decisions, data
        )

        # total student loss = standard + KD + feature
        loss = loss_standard + loss_kd + loss_feat

        status.update(distill_status)
        return loss, loss_adapt, status

    def forward_pass(self, data):
        b = data['search_images'].shape[1]  # n,b,c,h,w
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b, dim=0)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b, dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b, dim=0)

        out_list_student = []
        out_list_teacher = []
        all_adapt_logits = []
        all_adapt_decisions = []

        neck_h_state_student = [None] * self.cfg.MODEL.NECK.N_LAYERS
        neck_h_state_teacher = [None] * self.cfg.MODEL.NECK.N_LAYERS

        for i in range(len(search_list)):
            search_i_list = [search_list[i]]

            # --- Teacher forward (no grad) ---
            with torch.no_grad():
                enc_opt_t = self.net_teacher(
                    template_list=template_list, search_list=search_i_list,
                    template_anno_list=template_anno_list, mode='encoder'
                )
                encoder_out_t, neck_out_t, neck_h_state_teacher = self.net_teacher(
                    enc_opt=enc_opt_t, neck_h_state=neck_h_state_teacher, mode='neck'
                )
                outputs_t = self.net_teacher(feature=neck_out_t, mode='decoder')

            # --- Student forward ---
            enc_opt_s = self.net(
                template_list=template_list, search_list=search_i_list,
                template_anno_list=template_anno_list, mode='encoder'
            )
            encoder_out_s, neck_out_s, neck_h_state_student = self.net(
                enc_opt=enc_opt_s, neck_h_state=neck_h_state_student, mode='neck'
            )
            outputs_s = self.net(feature=neck_out_s, mode='decoder')

            # --- Adaptive network ---
            teacher_feat_list = [enc_opt_t]
            student_feat_list = [enc_opt_s]

            if hasattr(self.net, 'adjust_layers') and self.net.adjust_layers is not None:
                student_feat_aligned = [self.net.adjust_layers[j](student_feat_list[j])
                                        for j in range(len(student_feat_list))]
            else:
                student_feat_aligned = student_feat_list

            adapt_logits = self.adaptive_net(teacher_feat_list, student_feat_list)
            adapt_decisions_i = [gumbel_softmax(logit) for logit in adapt_logits]

            # Store outputs
            out_list_student.append({
                'outputs': outputs_s,
                'enc_feat': enc_opt_s,
                'enc_feat_aligned': student_feat_aligned[0] if student_feat_aligned else enc_opt_s,
                'neck_out': neck_out_s,
            })
            out_list_teacher.append({
                'outputs': outputs_t,
                'enc_feat': enc_opt_t,
                'neck_out': neck_out_t,
            })
            all_adapt_logits.append(adapt_logits)
            all_adapt_decisions.append(adapt_decisions_i)

        return out_list_student, out_list_teacher, all_adapt_logits, all_adapt_decisions

    def compute_losses(self, pred_list, gt_dict, return_status=True):
        """Compute standard tracking losses (same as MCITrackActor)."""
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda()
        gt_gaussian_maps_list = generate_heatmap(
            gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.ENCODER.STRIDE
        )

        for i in range(len(pred_list)):
            pred_dict = pred_list[i]['outputs']
            gt_bbox = gt_dict['search_anno'][i]
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)

            # Get boxes
            pred_boxes = pred_dict['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat(
                (1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)

            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
            # compute location loss
            if 'score_map' in pred_dict:
                location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)

            loss = (self.loss_weight['giou'] * giou_loss +
                    self.loss_weight['l1'] * l1_loss +
                    self.loss_weight['focal'] * location_loss)
            total_loss += loss

            if return_status:
                mean_iou = iou.detach().mean()
                status = {
                    f"{i}frame_Loss/total": loss.item(),
                    f"{i}frame_Loss/giou": giou_loss.item(),
                    f"{i}frame_Loss/l1": l1_loss.item(),
                    f"{i}frame_Loss/location": location_loss.item(),
                    f"{i}frame_IoU": mean_iou.item()
                }
                total_status.update(status)

        if return_status:
            return total_loss, total_status
        else:
            return total_loss

    def compute_losses_distill(self, out_student, out_teacher, adapt_logits, adapt_decisions, data):
        """Compute distillation losses: KD (KL-div on score maps) + feature MSE."""
        temperature = self.cfg.TRAIN.TEMPERATURE
        kd_weight = self.cfg.TRAIN.KD_WEIGHT
        feat_weight = self.cfg.TRAIN.FEAT_WEIGHT

        total_kd_loss = torch.tensor(0., dtype=torch.float).cuda()
        total_feat_loss = torch.tensor(0., dtype=torch.float).cuda()
        total_adapt_loss = torch.tensor(0., dtype=torch.float).cuda()

        for i in range(len(out_student)):
            student_out = out_student[i]['outputs']
            teacher_out = out_teacher[i]['outputs']
            decisions = adapt_decisions[i]
            logits = adapt_logits[i]

            # --- KD loss on score maps ---
            if 'score_map' in student_out and 'score_map' in teacher_out:
                s_score = student_out['score_map']
                t_score = teacher_out['score_map']
                B = s_score.shape[0]
                s_flat = s_score.view(B, -1)
                t_flat = t_score.view(B, -1)
                log_s = F.log_softmax(s_flat / temperature, dim=-1)
                soft_t = F.softmax(t_flat / temperature, dim=-1)
                kd_per_sample = F.kl_div(log_s, soft_t, reduction='none').sum(dim=-1) * (temperature ** 2)

                if len(decisions) > 0:
                    mask = decisions[0][:, 0].detach()
                    kd_loss = (kd_per_sample * mask).mean()
                else:
                    kd_loss = kd_per_sample.mean()
                total_kd_loss += kd_weight * kd_loss

            # --- Feature MSE loss ---
            student_feat_aligned = out_student[i]['enc_feat_aligned']
            teacher_feat = out_teacher[i]['enc_feat']
            feat_mse_per_sample = F.mse_loss(student_feat_aligned, teacher_feat.detach(), reduction='none').mean(dim=(1, 2))

            if len(decisions) > 0:
                mask = decisions[0][:, 0].detach()
                feat_loss = (feat_mse_per_sample * mask).mean()
            else:
                feat_loss = feat_mse_per_sample.mean()
            total_feat_loss += feat_weight * feat_loss

            # --- Adaptive network loss ---
            with torch.no_grad():
                s_boxes = box_cxcywh_to_xyxy(student_out['pred_boxes']) # (B, Q, 4)
                t_boxes = box_cxcywh_to_xyxy(teacher_out['pred_boxes']) # (B, Q, 4)
                gt_bbox = data['search_anno'][i] # (B, 4)
                gt_boxes = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0) # (B, 4)
                
                B, Q, _ = s_boxes.shape
                # Match each query in the whole batch to its corresponding sample's ground truth
                gt_boxes_rep = gt_boxes.unsqueeze(1).repeat(1, Q, 1).view(-1, 4)
                s_boxes_flat = s_boxes.view(-1, 4)
                t_boxes_flat = t_boxes.view(-1, 4)
                
                iou_s_vec, _ = box_iou(s_boxes_flat, gt_boxes_rep)
                iou_t_vec, _ = box_iou(t_boxes_flat, gt_boxes_rep)
                
                # In this repo, box_iou is element-wise (N,), so no .diag() is needed
                iou_s = iou_s_vec.view(B, Q).max(dim=1)[0]
                iou_t = iou_t_vec.view(B, Q).max(dim=1)[0]

                hard_target = (iou_t > iou_s).long() # (B,)

            # Train adaptive net with raw logits
            if len(logits) > 0:
                for logit in logits:
                    adapt_ce = F.cross_entropy(logit, hard_target)
                    total_adapt_loss += adapt_ce

        distill_status = {
            "Loss/kd": total_kd_loss.item(),
            "Loss/feat": total_feat_loss.item(),
            "Loss/adapt": total_adapt_loss.item(),
        }

        return total_kd_loss, total_feat_loss, total_adapt_loss, distill_status
