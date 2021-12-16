import torch
import torch.nn as nn
from utils import l2_loss, relative_to_abs, from_abs_to_social
from torch.nn.functional import cross_entropy


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) == 2:
            features=torch.unsqueeze(features, 1)
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.contrastive_loss = SupConLoss()
    
    def forward(self, model, train_iter, pretrain_iter, train_dataset, pretrain_dataset, training_step, args, stage):
        assert(training_step in ['P3', 'P4', 'P5', 'P6'])
        batch_loss = []
        env_embeddings, label_embeddings = [], [] # to store the low dim feat space for contrastive style loss, and their labels
        pred_embeddings = []  # store all the predictions for each env, to use for consistency loss
        ped_tot = torch.zeros(1).cuda()

        # COMPUTE LOSS ON EACH OF THE ENVIRONMENTS
        for env_iter, env_name in zip(train_iter, train_dataset['names']):
            try:
                batch = next(env_iter)
            except StopIteration:
                raise RuntimeError()

            # transfer batch     
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, _, obs_traj_rel, fut_traj_rel, seq_start_end, _, _) = batch
            ped_tot += fut_traj_rel.shape[1]
            scale = torch.tensor(1.).cuda().requires_grad_()

            # loss of style encoder only
            if training_step == 'P4':
                low_dim = model(batch, 'P4')
                env_embeddings.append(low_dim)
                label_embeddings.append(torch.tensor(train_dataset['labels'][env_name]))
                continue

            # compute model output
            if training_step == 'P6':
                fut_pred_rel, low_dim = model(batch, training_step)
                env_embeddings.append(low_dim)
                label_embeddings.append(torch.tensor(train_dataset['labels'][env_name]))
                if args.styleconsistency: pred_embeddings.append(fut_pred_rel)

            else:
                fut_pred_rel = model(batch, training_step)

            # compute loss between output and future
            l2_loss_rel = torch.stack([
                l2_loss(fut_pred_rel * scale, fut_traj_rel, mode="raw")
            ], dim=1)

            # empirical risk (ERM classic loss)
            loss_sum_even, loss_sum_odd = self.erm_loss(l2_loss_rel, seq_start_end, fut_traj_rel.shape[0])
            single_env_loss = loss_sum_even + loss_sum_odd
            
            # invariance constraint (IRM)
            if training_step in ['P1', 'P2', 'P3'] and args.irm and stage == 'training':
               single_env_loss += self.irm_loss(loss_sum_even, loss_sum_odd, scale, args)

            batch_loss.append(single_env_loss)


        # COMPUTE THE TOTAL LOSS ON ALL ENVIRONMENTS
        loss = torch.zeros(()).cuda()

        # content loss
        if training_step in ['P3', 'P5', 'P6']:
            batch_loss = torch.stack(batch_loss)
            loss += batch_loss.sum()

        # variance risk extrapolation for content
        if training_step ==  'P3' and args.vrex:
            loss += batch_loss.var() * args.vrex
        

        # style contrastive loss
        if stage=='training' and (training_step == 'P4' or (training_step == 'P6' and args.contrastive)):
            
            # if finetuning, need to add the low dim latent spaces of pretraining environments
            if args.finetune != '' and pretrain_iter and pretrain_dataset and training_step == 'P6' and args.styleconsistency == 0:
                self.add_feat_spaces_pretraining(model, env_embeddings, label_embeddings, pretrain_iter, pretrain_dataset)

            assert len(label_embeddings) > 1,  'Cannot train contrastive learning with only one label'

            # set to contrastive value if both loss (step 6), to 1 if this is the only loss (step 4)
            factor = args.contrastive if training_step == 'P6' else 1 
            loss += self.contrastive_loss(torch.stack(env_embeddings), torch.stack(label_embeddings)) * factor
        

        # style consistency loss: prediction needs to have the style of the env
        if training_step in ['P6'] and args.styleconsistency:

            cons_embeddings = [t.detach() for t in env_embeddings]
            cons_labels = [t.detach() for t in label_embeddings] * 2
            
            for pred_fut_rel  in pred_embeddings:    
                pred_fut = relative_to_abs(pred_fut_rel, obs_traj[-1, :, :2])
                pred_full_path = torch.cat((obs_traj, pred_fut))
                social_encoded_pred = from_abs_to_social(pred_full_path)
                low_dim = model.style_encoder(social_encoded_pred, 'low')
                cons_embeddings.append(low_dim)

            loss += self.contrastive_loss(torch.stack(cons_embeddings), torch.stack(cons_labels)) * args.styleconsistency

        return loss, ped_tot


    def add_feat_spaces_pretraining(self, model, env_embeddings, label_embeddings, pretrain_iter, pretrain_dataset):
        """ Add one random batch of styles of each pretrain environment """
        for env_iter, env_name in zip(pretrain_iter, pretrain_dataset['names']):
            try:
                batch = next(env_iter)
            except StopIteration:
                raise RuntimeError()
            batch = [tensor.cuda() for tensor in batch]
            env_embeddings.append(model(batch, 'P4'))
            label_embeddings.append(torch.tensor(pretrain_dataset['labels'][env_name]))

    def erm_loss(self, l2_loss_rel, seq_start_end, length_fut):
        loss_sum_even, loss_sum_odd = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        even = True
        for start, end in seq_start_end.data:
            _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
            _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [best_k elements]
            _l2_loss_rel = torch.min(_l2_loss_rel) / ((length_fut) * (end - start))
            if even == True:
                loss_sum_even += _l2_loss_rel
                even = False
            else:
                loss_sum_odd += _l2_loss_rel
                even = True
        return loss_sum_even, loss_sum_odd

    def irm_loss(self, loss_sum_even, loss_sum_odd, scale, args):
        if args.unbiased:
            g1 = torch.autograd.grad(loss_sum_even, [scale], create_graph=True)[0]
            g2 = torch.autograd.grad(loss_sum_odd, [scale], create_graph=True)[0]
            inv_constr = g1 * g2
            additional_loss = inv_constr * args.irm
        else:
            grad = torch.autograd.grad(loss_sum_even + loss_sum_odd, [scale], create_graph=True)[0]
            inv_constr = torch.sum(grad ** 2)
            additional_loss = inv_constr * args.irm
        return additional_loss

criterion = CustomLoss().cuda()

def standard_style_loss(output_classifier, label):
    # compute the good loss according to classification
    t = torch.tensor([label] * output_classifier.shape[0]).cuda()
    single_env_loss = cross_entropy(output_classifier, t)
    return single_env_loss

