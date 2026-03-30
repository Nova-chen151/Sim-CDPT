import copy
import os
import torch
import lightning.pytorch as pl
from .modules import Encoder, Denoiser, GoalPredictor
from .utils import DDPM_Sampler
from .model_utils import inverse_kinematics, roll_out, batch_transform_trajs_to_global_frame
from torch.nn.functional import smooth_l1_loss, cross_entropy
from .distill import (
    denoise_kl_distill_loss,
    encoder_feature_distillation_loss,
    mi_distill_loss,
)

# from geomloss import SamplesLoss

class CDPT(pl.LightningModule):
    """
    Transferring Causal Driving Patterns for Generalizable Traffic Simulation with Diffusion-Based Distillation
    """

    def __init__(
            self,
            cfg: dict,
    ):
        """
        CDPT with optional same-architecture teacher (another CDPT/VBD-style ckpt with encoder/denoiser/predictor).

        Args:
            cfg (dict): Model configuration parameters.
        """
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self._future_len = cfg['future_len']
        self._agents_len = cfg['agents_len']
        self._action_len = cfg['action_len']
        self._diffusion_steps = cfg['diffusion_steps']
        self._encoder_layers = cfg['encoder_layers']
        self._encoder_version = cfg.get('encoder_version', 'v1')
        self._action_mean = cfg['action_mean']
        self._action_std = cfg['action_std']

        self._train_encoder = cfg.get('train_encoder', True)
        self._train_denoiser = cfg.get('train_denoiser', True)
        self._train_predictor = cfg.get('train_predictor', True)
        self._with_predictor = cfg.get('with_predictor', True)
        self._prediction_type = cfg.get('prediction_type', 'sample')
        self._schedule_type = cfg.get('schedule_type', 'cosine')
        self._embeding_dim = cfg.get('embeding_dim', 5)

        self.encoder = Encoder(self._encoder_layers, version=self._encoder_version)
        self.denoiser = Denoiser(
            future_len=self._future_len,
            action_len=self._action_len,
            agents_len=self._agents_len,
            steps=self._diffusion_steps,
            input_dim=self._embeding_dim,
        )
        if self._with_predictor:
            self.predictor = GoalPredictor(
                future_len=self._future_len,
                agents_len=self._agents_len,
                action_len=self._action_len,
            )
        else:
            self.predictor = None
            self._train_predictor = False

        self.noise_scheduler = DDPM_Sampler(
            steps=self._diffusion_steps,
            schedule=self._schedule_type,
            s=cfg.get('schedule_s', 0.0),
            e=cfg.get('schedule_e', 1.0),
            tau=cfg.get('schedule_tau', 1.0),
            scale=cfg.get('schedule_scale', 1.0),
        )

        self.register_buffer('action_mean', torch.tensor(self._action_mean))
        self.register_buffer('action_std', torch.tensor(self._action_std))

        # Teacher is only for training-time distillation; set use_teacher: false for deployment / inference.
        self._use_teacher = cfg.get('use_teacher', True)
        self._save_teacher_in_checkpoint = cfg.get('save_teacher_in_checkpoint', True)
        self.teacher_model = None
        if self._use_teacher:
            teacher_path = (
                cfg.get('teacher_checkpoint_path')
                or cfg.get('teacher_model_path')
                or './train_log/baseline/epoch=19.ckpt'
            )
            if not teacher_path or not os.path.isfile(teacher_path):
                raise FileNotFoundError(
                    f"use_teacher=True but teacher checkpoint not found: {teacher_path!r}. "
                    "Set teacher_checkpoint_path / teacher_model_path, or use_teacher: false for inference-only."
                )
            self.teacher_model = _load_teacher_from_checkpoint(teacher_path, cfg)

        self._distill_weight_start = cfg.get('distill_weight_start', 0.8)
        self._distill_weight_end = cfg.get('distill_weight_end', 0.2)
        self._anneal_steps = cfg.get('anneal_steps', 40000)
        self._my_global_step = 0

    ################### Training Setup ###################
    def configure_optimizers(self):
        '''
        This function is called by Lightning to create the optimizer and learning rate scheduler.
        '''
        if not self._train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not self._train_denoiser:
            for param in self.denoiser.parameters():
                param.requires_grad = False
        if self._with_predictor and (not self._train_predictor):
            for param in self.predictor.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight_decay']
        )

        lr_warmpup_step = self.cfg['lr_warmup_step']
        lr_step_freq = self.cfg['lr_step_freq']
        lr_step_gamma = self.cfg['lr_step_gamma']

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n

            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step,
                lr_warmpup_step,
                lr_step_freq,
                lr_step_gamma,
            )
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, inputs, noised_actions_normalized, diffusion_step):
        """
        Forward pass, returns student and teacher model outputs.

        Args:
            inputs: Input data.
            noised_actions_normalized: Normalized noisy actions.
            diffusion_step: Current diffusion step.

        Returns:
            tuple: student_output_dict
        """
        output_dict = {}
        encoder_outputs = self.encoder(inputs)

        if self._train_denoiser:
            student_denoise_outputs, _ = self.forward_denoiser(
                encoder_outputs, noised_actions_normalized, diffusion_step
            )
            output_dict.update(student_denoise_outputs)

        if self._train_predictor:
            student_predictor_outputs, _ = self.forward_predictor(encoder_outputs)
            output_dict.update(student_predictor_outputs)

        return output_dict

    def forward_denoiser(self, encoder_outputs, noised_actions_normalized, diffusion_step):
        """
        Denoiser forward pass, returns student and teacher model outputs.

        Args:
            encoder_outputs: Encoder outputs.
            noised_actions_normalized: Normalized noisy actions.
            diffusion_step: Current diffusion step.

        Returns:
            tuple: (student_outputs, teacher_outputs).
        """
        noised_actions = self.unnormalize_actions(noised_actions_normalized)
        denoiser_output = self.denoiser(encoder_outputs, noised_actions, diffusion_step)
        denoised_actions_normalized = self.noise_scheduler.q_x0(
            denoiser_output, diffusion_step,
            noised_actions_normalized,
            prediction_type=self._prediction_type
        )
        current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'

        denoised_actions = self.unnormalize_actions(denoised_actions_normalized)
        denoised_trajs = roll_out(current_states, denoised_actions,
                                  action_len=self.denoiser._action_len, global_frame=True)

        student_outputs = {
            'denoiser_output': denoiser_output,
            'denoised_actions_normalized': denoised_actions_normalized,
            'denoised_actions': denoised_actions,
            'denoised_trajs': denoised_trajs,
        }

        teacher_outputs = None
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_denoiser_output = self.teacher_model.denoiser(encoder_outputs, noised_actions, diffusion_step)
                teacher_denoised_actions_normalized = self.teacher_model.noise_scheduler.q_x0(
                    teacher_denoiser_output, diffusion_step,
                    noised_actions_normalized,
                    prediction_type=self._prediction_type
                )
                teacher_denoised_actions = self.teacher_model.unnormalize_actions(teacher_denoised_actions_normalized)
                teacher_denoised_trajs = roll_out(
                    current_states, teacher_denoised_actions,
                    action_len=self.teacher_model.denoiser._action_len, global_frame=True
                )
                teacher_outputs = {
                    'denoiser_output': teacher_denoiser_output,
                    'denoised_actions_normalized': teacher_denoised_actions_normalized,
                    'denoised_actions': teacher_denoised_actions,
                    'denoised_trajs': teacher_denoised_trajs,
                }

        return student_outputs, teacher_outputs

    def forward_predictor(self, encoder_outputs):
        """
        Predictor forward pass, returns student and teacher model outputs.

        Args:
            encoder_outputs: Encoder outputs.

        Returns:
            tuple: (student_outputs, teacher_outputs).
        """
        goal_actions_normalized, goal_scores = self.predictor(encoder_outputs)
        current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'

        goal_actions = self.unnormalize_actions(goal_actions_normalized)
        goal_trajs = roll_out(current_states[:, :, None, :], goal_actions,
                              action_len=self.predictor._action_len, global_frame=True)

        student_outputs = {
            'goal_actions_normalized': goal_actions_normalized,
            'goal_actions': goal_actions,
            'goal_scores': goal_scores,
            'goal_trajs': goal_trajs,
        }

        teacher_outputs = None
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_goal_actions_normalized, teacher_goal_scores = self.teacher_model.predictor(encoder_outputs)
                teacher_goal_actions = self.teacher_model.unnormalize_actions(teacher_goal_actions_normalized)
                teacher_goal_trajs = roll_out(
                    current_states[:, :, None, :], teacher_goal_actions,
                    action_len=self.teacher_model.predictor._action_len, global_frame=True
                )
                teacher_outputs = {
                    'goal_actions_normalized': teacher_goal_actions_normalized,
                    'goal_actions': teacher_goal_actions,
                    'goal_scores': teacher_goal_scores,
                    'goal_trajs': teacher_goal_trajs,
                }

        return student_outputs, teacher_outputs

    def forward_and_get_loss(self, batch, prefix='', debug=False):
        """
        Forward pass and compute loss with knowledge distillation.

        Args:
            batch: Input batch data.
            prefix: Prefix for loss keys.
            debug: Whether to enable debug mode.

        Returns:
            tuple: (total_loss, log_dict) or (total_loss, log_dict, debug_outputs) if debug=True.
        """
        agents_future = batch['agents_future'][:, :self._agents_len]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)
        agents_interested = batch['agents_interested'][:, :self._agents_len]
        anchors = batch['anchors'][:, :self._agents_len]

        gt_actions, gt_actions_valid = inverse_kinematics(
            agents_future,
            agents_future_valid,
            dt=0.1,
            action_len=self._action_len
        )
        gt_actions_normalized = self.normalize_actions(gt_actions)
        B, A, T, D = gt_actions_normalized.shape

        log_dict = {}
        debug_outputs = {}
        total_loss = 0

        ############## Run Encoder ##############
        encoder_outputs = self.encoder(batch)
        distill_weight, other_weight = self.get_annealed_weights()
        student_attention_values = encoder_outputs['attention_values']

        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_encoder_outputs = self.teacher_model.encoder(batch)
                teacher_attention_values = teacher_encoder_outputs['attention_values']
            feature_distillation_loss = encoder_feature_distillation_loss(
                student_attention_values, teacher_attention_values
            )
            total_loss += 5.0 * feature_distillation_loss
        else:
            feature_distillation_loss = torch.tensor(0.0, device=agents_future.device)

        ############### Denoise #################
        if self._train_denoiser:
            diffusion_steps = torch.randint(
                0, self.noise_scheduler.num_steps, (B,),
                device=agents_future.device
            ).long().unsqueeze(-1).repeat(1, A).view(B, A, 1, 1)

            noise = torch.randn(B, A, T, D).type_as(agents_future)
            noised_action_normalized = self.noise_scheduler.add_noise(gt_actions_normalized, noise, diffusion_steps)

            student_denoise_outputs, teacher_denoise_outputs = self.forward_denoiser(
                encoder_outputs, noised_action_normalized, diffusion_steps.view(B, A)
            )

            debug_outputs.update(student_denoise_outputs)
            debug_outputs['noise'] = noise
            debug_outputs['diffusion_steps'] = diffusion_steps

            denoised_trajs = student_denoise_outputs['denoised_trajs']

            if self._prediction_type == 'sample':
                state_loss_mean, yaw_loss_mean = self.denoise_loss(
                    denoised_trajs,
                    agents_future,
                    agents_future_valid,
                    agents_interested
                )
                denoise_loss = state_loss_mean + yaw_loss_mean

                _, diffusion_loss = self.noise_scheduler.get_noise(
                    x_0=student_denoise_outputs['denoised_actions_normalized'],
                    x_t=noised_action_normalized,
                    timesteps=diffusion_steps,
                    gt_noise=noise,
                )

                log_dict.update({
                    prefix + 'state_loss': state_loss_mean.item(),
                    prefix + 'yaw_loss': yaw_loss_mean.item(),
                    prefix + 'diffusion_loss': diffusion_loss.item()
                })

            elif self._prediction_type == 'error':
                denoiser_output = student_denoise_outputs['denoiser_output']
                denoise_loss = torch.nn.functional.mse_loss(
                    denoiser_output, noise, reduction='mean'
                )
                log_dict.update({
                    prefix + 'diffusion_loss': denoise_loss.item(),
                })

            elif self._prediction_type == 'mean':
                denoise_loss = self.action_loss(
                    student_denoise_outputs['denoised_actions_normalized'],
                    gt_actions_normalized,
                    gt_actions_valid, agents_interested
                )
                log_dict.update({
                    prefix + 'action_loss': denoise_loss.item(),
                })
            else:
                raise ValueError('Invalid prediction type')

            ################# Denoising Distillation Loss (training + teacher only) ################
            if self.teacher_model is not None and teacher_denoise_outputs is not None:
                mi_loss = mi_distill_loss(
                    student_denoise_outputs['denoised_actions_normalized'],
                    teacher_denoise_outputs['denoised_actions_normalized'],
                    scoring_fn='cosine',
                )
                denoise_distill_loss_KL = denoise_kl_distill_loss(
                    student_denoise_outputs['denoised_actions_normalized'],
                    teacher_denoise_outputs['denoised_actions_normalized'],
                )

                distill_loss_components = (
                    denoise_distill_loss_KL * 1.0 +
                    feature_distillation_loss * 2.0 +
                    mi_loss * 1.0
                )
                total_denoise_loss = denoise_loss * other_weight + distill_weight * distill_loss_components
                denoise_distill_loss_KL_val = denoise_distill_loss_KL.item()
                total_distill_val = distill_loss_components.item()
            else:
                denoise_distill_loss_KL_val = 0.0
                total_distill_val = 0.0
                total_denoise_loss = denoise_loss

            total_loss += total_denoise_loss

            denoise_ade, denoise_fde = self.calculate_metrics_denoise(
                denoised_trajs, agents_future, agents_future_valid, agents_interested, 8
            )

            log_dict.update({
                prefix + 'denoise_loss': denoise_loss.item(),
                prefix + 'denoise_distill_loss_KL': denoise_distill_loss_KL_val,
                prefix + 'feature_distillation_loss': feature_distillation_loss.item(),
                prefix + 'total_distill_loss': total_distill_val,
                prefix + 'denoise_ADE': denoise_ade,
                prefix + 'denoise_FDE': denoise_fde,
            })

        if self._train_predictor:
            student_goal_outputs, teacher_goal_outputs = self.forward_predictor(encoder_outputs)
            debug_outputs.update(student_goal_outputs)

            goal_trajs = student_goal_outputs['goal_trajs']
            goal_scores = student_goal_outputs['goal_scores']

            goal_loss_mean, score_loss_mean = self.goal_loss(
                goal_trajs, goal_scores, agents_future,
                agents_future_valid, anchors, agents_interested
            )
            pred_loss = goal_loss_mean + 0.05 * score_loss_mean

            total_loss += 1.0 * pred_loss * other_weight

            pred_ade, pred_fde = self.calculate_metrics_predict(
                goal_trajs, agents_future, agents_future_valid, agents_interested, 8
            )

            log_dict.update({
                prefix + 'goal_loss': goal_loss_mean.item(),
                prefix + 'score_loss': score_loss_mean.item(),
                prefix + 'pred_ADE': pred_ade,
                prefix + 'pred_FDE': pred_fde,
            })

        log_dict[prefix + 'loss'] = total_loss.item()

        if debug:
            return total_loss, log_dict, debug_outputs
        else:
            return total_loss, log_dict

    def training_step(self, batch, batch_idx):
        self._my_global_step += 1
        
        loss, log_dict = self.forward_and_get_loss(batch, prefix='train/')
        
        distill_weight, other_weight = self.get_annealed_weights()
        log_dict.update({
            'train/distill_weight': distill_weight.item(),
            'train/other_weight': other_weight.item(),
            'train/global_step': self._my_global_step,
        })
        
        self.log_dict(log_dict,
                      on_step=False, on_epoch=True, sync_dist=True,
                      prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.forward_and_get_loss(batch, prefix='val/')
        self.log_dict(log_dict,
                      on_step=False, on_epoch=True, sync_dist=True,
                      prog_bar=True)
        return loss

    def on_save_checkpoint(self, checkpoint):
        """Drop teacher weights from ckpt when save_teacher_in_checkpoint is false (smaller student-only bundles)."""
        if self._save_teacher_in_checkpoint or self.teacher_model is None:
            return
        sd = checkpoint.get('state_dict')
        if not sd:
            return
        for k in list(sd.keys()):
            if k.startswith('teacher_model.'):
                del sd[k]

    @classmethod
    def load_for_inference(cls, checkpoint_path, map_location=None, **model_init_kwargs):
        """
        Load student weights only: does not construct or load the teacher (true deployment distillation).
        Use strict=False so checkpoints that still contain teacher_model.* keys are ignored.
        """
        try:
            ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=map_location)
        hp = ckpt.get('hyper_parameters', {})
        raw_cfg = hp.get('cfg')
        if raw_cfg is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path!r} has no hyper_parameters['cfg']; "
                "cannot rebuild model. Instantiate with cfg and load_state_dict manually."
            )
        cfg = copy.deepcopy(raw_cfg) if isinstance(raw_cfg, dict) else copy.deepcopy(dict(raw_cfg))
        cfg['use_teacher'] = False
        model = cls(cfg, **model_init_kwargs)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        return model

    ################### Loss function ###################
    def denoise_loss(
            self, denoised_trajs,
            agents_future, agents_future_valid,
            agents_interested
    ):
        """
        Calculates the denoise loss for the denoised actions and trajectories.

        Args:
            denoised_trajs (torch.Tensor): Denoised trajectories tensor of shape [B, A, T, C].
            agents_future (torch.Tensor): Future agent positions tensor of shape [B, A, T, 3].
            agents_future_valid (torch.Tensor): Future agent validity tensor of shape [B, A, T].
            agents_interested (torch.Tensor): Interested agents tensor of shape [B, A].

        Returns:
            state_loss_mean (torch.Tensor): Mean state loss.
            yaw_loss_mean (torch.Tensor): Mean yaw loss.
        """

        agents_future = agents_future[..., 1:, :3]
        future_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)

        # Calculate State Loss
        state_loss = smooth_l1_loss(denoised_trajs[..., :2], agents_future[..., :2], reduction='none').sum(
            -1)
        yaw_error = (denoised_trajs[..., 2] - agents_future[..., 2])
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        yaw_loss = torch.abs(yaw_error)

        # Filter out the invalid state
        state_loss = state_loss * future_mask
        yaw_loss = yaw_loss * future_mask

        # Calculate the mean loss
        state_loss_mean = state_loss.sum() / future_mask.sum()
        yaw_loss_mean = yaw_loss.sum() / future_mask.sum()

        return state_loss_mean, yaw_loss_mean

    def action_loss(
            self, actions, actions_gt, actions_valid, agents_interested
    ):
        """
        Calculates the loss for action prediction.

        Args:
            actions (torch.Tensor): Tensor of shape [B, A, T, 2] representing predicted actions.
            actions_gt (torch.Tensor): Tensor of shape [B, A, T, 2] representing ground truth actions.
            actions_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of actions.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            action_loss_mean (torch.Tensor): Mean action loss.
        """
        # Get Mask
        action_mask = actions_valid * (agents_interested[..., None] > 0)

        # Calculate the action loss
        action_loss = smooth_l1_loss(actions, actions_gt, reduction='none').sum(-1)
        action_loss = action_loss * action_mask

        # Calculate the mean loss
        action_loss_mean = action_loss.sum() / action_mask.sum()

        return action_loss_mean

    def goal_loss(
            self, trajs, scores, agents_future,
            agents_future_valid, anchors,
            agents_interested
    ):
        """
        Calculates the loss for trajectory prediction.

        Args:
            trajs (torch.Tensor): Tensor of shape [B*A, Q, T, 3] representing predicted trajectories.
            scores (torch.Tensor): Tensor of shape [B*A, Q] representing predicted scores.
            agents_future (torch.Tensor): Tensor of shape [B, A, T, 3] representing future agent states.
            agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of future agent states.
            anchors (torch.Tensor): Tensor of shape [B, A, Q, 2] representing anchor points.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            traj_loss_mean (torch.Tensor): Mean trajectory loss.
            score_loss_mean (torch.Tensor): Mean score loss.
        """
        # Convert Anchor to Global Frame
        current_states = agents_future[:, :, 0, :3]
        anchors_global = batch_transform_trajs_to_global_frame(anchors, current_states)
        num_batch, num_agents, num_query, _ = anchors_global.shape

        # Get Mask
        traj_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)

        # Flatten batch and agents
        goal_gt = agents_future[:, :, -1:, :2].flatten(0, 1)
        trajs_gt = agents_future[:, :, 1:, :3].flatten(0, 1)
        trajs = trajs.flatten(0, 1)[..., :3]
        anchors_global = anchors_global.flatten(0, 1)

        # Find the closest anchor
        idx_anchor = torch.argmin(torch.norm(anchors_global - goal_gt, dim=-1), dim=-1)

        # For agents that do not have valid end point, use the minADE
        dist = torch.norm(trajs[:, :, :, :2] - trajs_gt[:, None, :, :2], dim=-1)
        dist = dist * traj_mask.flatten(0, 1)[:, None, :]
        idx = torch.argmin(dist.mean(-1), dim=-1)

        # Select trajectory
        idx = torch.where(agents_future_valid[..., -1].flatten(0, 1), idx_anchor, idx)
        trajs_select = trajs[torch.arange(num_batch * num_agents), idx]

        # Calculate the trajectory loss
        traj_loss = smooth_l1_loss(trajs_select, trajs_gt, reduction='none').sum(-1)
        traj_loss = traj_loss * traj_mask.flatten(0, 1)

        # Calculate the score loss
        scores = scores.flatten(0, 1)
        score_loss = cross_entropy(scores, idx, reduction='none')
        score_loss = score_loss * (agents_interested.flatten(0, 1) > 0)

        # Calculate the mean loss
        traj_loss_mean = traj_loss.sum() / traj_mask.sum()
        score_loss_mean = score_loss.sum() / (agents_interested > 0).sum()

        return traj_loss_mean, score_loss_mean

    @torch.no_grad()
    def calculate_metrics_denoise(self,
                                  denoised_trajs, agents_future, agents_future_valid,
                                  agents_interested, top_k=None
                                  ):
        """
        Calculates the denoising metrics for the predicted trajectories.

        Args:
            denoised_trajs (torch.Tensor): Denoised trajectories of shape [B, A, T, 2].
            agents_future (torch.Tensor): Ground truth future trajectories of agents of shape [B, A, T, 2].
            agents_future_valid (torch.Tensor): Validity mask for future trajectories of agents of shape [B, A, T].
            agents_interested (torch.Tensor): Interest mask for agents of shape [B, A].
            top_k (int, optional): Number of top agents to consider. Defaults to None.

        Returns:
            Tuple[float, float]: A tuple containing the denoising ADE (Average Displacement Error) and FDE (Final Displacement Error).
        """

        if not top_k:
            top_k = self._agents_len

        pred_traj = denoised_trajs[:, :top_k, :, :2]
        gt = agents_future[:, :top_k, 1:, :2]
        gt_mask = (agents_future_valid[:, :top_k, 1:] & (
                    agents_interested[:, :top_k, None] > 0)).bool()

        denoise_mse = torch.norm(pred_traj - gt, dim=-1)
        denoise_ADE = denoise_mse[gt_mask].mean()
        denoise_FDE = denoise_mse[..., -1][gt_mask[..., -1]].mean()

        return denoise_ADE.item(), denoise_FDE.item()

    @torch.no_grad()
    def calculate_metrics_predict(self,
                                  goal_trajs, agents_future, agents_future_valid,
                                  agents_interested, top_k=None
                                  ):
        """
        Calculates the metrics for predicting goal trajectories.

        Args:
            goal_trajs (torch.Tensor): Tensor of shape [B, A, Q, T, 2] representing the goal trajectories.
            agents_future (torch.Tensor): Tensor of shape [B, A, T, 2] representing the future trajectories of agents.
            agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing the validity of future trajectories.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing the interest level of agents.
            top_k (int, optional): The number of top agents to consider. Defaults to None.

        Returns:
            tuple: A tuple containing the goal Average Displacement Error (ADE) and goal Final Displacement Error (FDE).
        """

        if not top_k:
            top_k = self._agents_len
        goal_trajs = goal_trajs[:, :top_k, :, :, :2]
        gt = agents_future[:, :top_k, 1:, :2]
        gt_mask = (agents_future_valid[:, :top_k, 1:] & (
                    agents_interested[:, :top_k, None] > 0)).bool()

        goal_mse = torch.norm(goal_trajs - gt[:, :, None, :, :], dim=-1)
        goal_mse = goal_mse * gt_mask[..., None, :]
        best_idx = torch.argmin(goal_mse.sum(-1), dim=-1)

        best_goal_mse = goal_mse[torch.arange(goal_mse.shape[0])[:, None],
        torch.arange(goal_mse.shape[1])[None, :],
        best_idx]

        goal_ADE = best_goal_mse.sum() / gt_mask.sum()
        goal_FDE = best_goal_mse[..., -1].sum() / gt_mask[..., -1].sum()

        return goal_ADE.item(), goal_FDE.item()

    ################### Helper Functions ##############
    def batch_to_device(self, input_dict: dict, device: torch.device = 'cuda'):
        """
        Move the tensors in the input dictionary to the specified device.

        Args:
            input_dict (dict): A dictionary containing tensors to be moved.
            device (torch.device): The target device to move the tensors to.

        Returns:
            dict: The input dictionary with tensors moved to the specified device.
        """
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device)

        return input_dict
    
    def get_annealed_weights(self):
        """
        Returns:
            tuple: (distill_weight, other_weight)
        """
        if self._my_global_step >= self._anneal_steps:
            distill_weight = torch.tensor(self._distill_weight_end)
        else:
            progress = self._my_global_step / self._anneal_steps
            
            warmup_ratio = 0.1
            if progress < warmup_ratio:
                warmup_progress = progress / warmup_ratio
                distill_weight = torch.tensor(self._distill_weight_start * warmup_progress)
            else:
                anneal_progress = (progress - warmup_ratio) / (1 - warmup_ratio)
                distill_weight = torch.tensor(self._distill_weight_end + \
                                (self._distill_weight_start - self._distill_weight_end) * \
                                (1 + torch.cos(torch.tensor(anneal_progress * torch.pi))) / 2)
        
        distill_weight = torch.clamp(distill_weight, min=0.0, max=1.0)
        
        other_weight = 1.0 - distill_weight * 0.8
        
        return distill_weight, other_weight
    
    def reset_global_step(self):
        """
        Reset global step for restarting training or debugging
        """
        self._my_global_step = 0
        print(f"Reset global step to: {self._my_global_step}")
    
    def get_anneal_info(self):
        """
        Get current annealing information
        Returns:
            dict: Dictionary containing annealing related information
        """
        distill_weight, other_weight = self.get_annealed_weights()
        progress = self._my_global_step / self._anneal_steps if self._anneal_steps > 0 else 0
        
        return {
            'global_step': self._my_global_step,
            'anneal_steps': self._anneal_steps,
            'progress': progress,
            'distill_weight': distill_weight.item(),
            'other_weight': other_weight.item(),
            'distill_weight_start': self._distill_weight_start,
            'distill_weight_end': self._distill_weight_end,
        }

    def normalize_actions(self, actions: torch.Tensor):  # Define action normalization method
        """
        Normalize the given actions using the mean and standard deviation.  # Document explanation for normalizing actions using mean and standard deviation

        Args:
            actions : The actions to be normalized.  # Action tensor to be normalized

        Returns:
            The normalized actions.  # Return normalized actions
        """
        return (actions - self.action_mean) / self.action_std  # Subtract mean from actions and divide by standard deviation to achieve normalization

    def unnormalize_actions(self, actions: torch.Tensor):  # Define action unnormalization method
        """
        Unnormalize the given actions using the stored action standard deviation and mean.  # Document explanation for unnormalizing actions using stored standard deviation and mean

        Args:
            actions: The normalized actions to be unnormalized.  # Normalized action tensor to be unnormalized

        Returns:
             The unnormalized actions.  # Return unnormalized actions
        """
        return actions * self.action_std + self.action_mean  # Multiply actions by standard deviation and add mean to achieve unnormalization


def _load_teacher_from_checkpoint(teacher_path: str, cfg: dict) -> "CDPT":
    """Load a frozen teacher with the same module layout as CDPT (no nested teacher). Uses strict=False for legacy VBD-style ckpts."""
    teacher_cfg = copy.deepcopy(cfg)
    teacher_cfg["use_teacher"] = False
    try:
        ckpt = torch.load(teacher_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(teacher_path, map_location="cpu")
    teacher = CDPT(teacher_cfg)
    teacher.load_state_dict(ckpt["state_dict"], strict=False)
    teacher.eval()
    teacher.freeze()
    return teacher
