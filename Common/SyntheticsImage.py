import os
import torch

from ..D

class SyntheicsTrainer:
    def __init__(self, cfg, model_name='seg_train'):
        self.cfg = cfg
        self._model_name = model_name
        self._is_build = false

    @property
    def model_name(self):
        return self._model_name

    @property
    def is_main_process(self):
        return self.local_rank == 0

    def _build(self):
        self.local_rank = get_rank()
        self.world_size = torch.cuda.device_count()

        self._build_parameters()
        self._build_writes()
        self._build_dataset()

        self._build_network()
        self._build_solver()
        self._load_checkpoint()
        self._build_ddp_apex_training()
        self._build_loss()
        self._build_metric()

        self._is_build = true

    def _build_parameters(self):
        self.start_epoch = self.cfg.slover.start_epoch
        self.total_epoch = 10 if self.cfg.env.smoke_test else self.cfg.slover.total_epoch
        self.lr = self.cfg.slover.lr
        self.best_dice = 0

        if torch.cuda.is_available():
            self.is_cuda = true
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
        else:
            self.is_cuda = false
            self.device = torch.device('cpu', self.local_rank)

        self.is_apex_train = self.cfg.env.is_apex_train if self.is_cuda else false
        self.is_distributed_train = self.cfg.env.is_distributed_train if self.is_cuda and \
                                    self.world_size > 1 else false

    def _build_writes(self):
        self.base_save_dir = os.path.join(self.cfg.env.save_dir,
                                          self.cfg.env.exp_name + '_result'
                                          '_time-' + datetime.datetime.now().strftime('%y-%m-%d_%h-%m-%s'))
        if self.is_main_process and not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)

        # copy codes
        self.code_save_dir = os.path.join(self.base_save_dir, 'codes')
        if self.is_main_process:
            if not os.path.exists(self.code_save_dir):
                os.makedirs(self.code_save_dir)
            shutil.copytree(base_dir+'/baseseg', self.code_save_dir+'/baseseg')
            shutil.copytree(base_dir+'/common', self.code_save_dir+'/common')
            if self.cfg.data_loader.semi_dataset:
                shutil.copy('./semi_config.yaml', self.code_save_dir + '/semi_config.yaml')
            else:
                shutil.copy('./full_config.yaml', self.code_save_dir+'/full_config.yaml')
            shutil.copy('./run.py', self.code_save_dir+'/run.py')

        # log dir
        self.log_dir = os.path.join(self.base_save_dir, 'logs')
        if self.is_main_process and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if self.is_main_process:
            self.logger = get_logger(self.log_dir)
            self.logger.info('\n------------ train options -------------')
            self.logger.info(str(self.cfg.pretty_text))
            self.logger.info('-------------- end ----------------\n')

        # model dir
        self.model_dir = os.path.join(self.base_save_dir, 'models')
        if self.is_main_process and not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.save_weight_path = os.path.join(self.model_dir, 'best_model.pt')
        self.pretrain_model_path = self.cfg.model.weight_path

        # tensorboard writer
        if self.is_main_process:
            self.train_writer = summarywriter(log_dir=os.path.join(self.log_dir, 'train'))
            self.val_writer = summarywriter(log_dir=os.path.join(self.log_dir, 'val'))

    def _build_dataset(self):
        if self.cfg.data_loader.semi_dataset:
            train_dataset = semisegdataset(self.cfg, 'train')
        else:
            train_dataset = segdataset(self.cfg, 'train')
        val_dataset = segdataset(self.cfg, 'val')

        if self.is_distributed_train:
            self.train_sampler = torch.utils.data.distributed.distributedsampler(train_dataset)
            self.val_sampler = torch.utils.data.distributed.distributedsampler(val_dataset)
        else:
            self.train_sampler = none
            self.val_sampler = none
        self.num_worker = self.cfg.data_loader.batch_size + 2 if self.cfg.data_loader.num_worker < \
                              self.cfg.data_loader.batch_size + 2 else self.cfg.data_loader.num_worker

        self.train_loader = dataloaderx(
            dataset=train_dataset,
            batch_size=self.cfg.data_loader.batch_size,
            num_workers=self.num_worker,
            shuffle=true if self.train_sampler is none and not self.cfg.data_loader.semi_dataset else false,
            drop_last=false,
            pin_memory=true,
            sampler=self.train_sampler)
        self.val_loader = dataloaderx(
            dataset=val_dataset,
            batch_size=self.cfg.data_loader.batch_size,
            num_workers=self.num_worker,
            shuffle=false,
            drop_last=false,
            pin_memory=true,
            sampler=self.val_sampler)
        if self.is_main_process:
            self.logger.info('build data loader success!')

    def _build_network(self):
        self.model = get_network(self.cfg.model)
        self.model = self.model.to(self.device)

    def _build_solver(self):
        self.optimizer = get_optimizer(self.cfg.slover, self.model.parameters())
        self.lr_scheduler = get_lr_scheduler(self.cfg.slover, self.optimizer)

    def _build_ddp_apex_training(self):
        # set apex training
        if self.is_apex_train:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='o1')

        # set distribute training
        if not self.is_distributed_train and self.world_size > 1:
            self.model = torch.nn.dataparallel(self.model)
        elif self.is_apex_train and self.is_distributed_train:
            self.model = distributeddataparallel(self.model, delay_allreduce=true)
        elif self.is_distributed_train:
            self.model = torch.nn.parallel.distributeddataparallel(self.model,
                                                                   device_ids=[self.local_rank],
                                                                   output_device=self.local_rank)

    def _build_loss(self):
        self.train_loss_func = segloss(loss_func=self.cfg.slover.loss,
                                       loss_weight=self.cfg.slover.loss_weight,
                                       activation=self.cfg.model.activation,
                                       reduction='mean',
                                       num_label=self.cfg.model.num_class)
        self.val_loss_func = segloss(loss_func=self.cfg.slover.loss,
                                     loss_weight=self.cfg.slover.loss_weight,
                                     activation=self.cfg.model.activation,
                                     reduction='sum',
                                     num_label=self.cfg.model.num_class)

    def _build_metric(self):
        self.train_metric_func = get_metric(metric=self.cfg.slover.metric,
                                            activation=self.cfg.model.activation,
                                            reduction='sum')
        self.val_metric_func = get_metric(metric=self.cfg.slover.metric,
                                          activation=self.cfg.model.activation,
                                          reduction='sum')

    def run(self):
        if not self._is_build:
            self._build()
        if self.is_main_process:
            run_start_time = get_time(self.is_cuda)
            self.logger.info(f'preprocess parallels: {self.num_worker}')
            self.logger.info(f'train samples per epoch: {len(self.train_loader)}')
            self.logger.info(f'val samples per epoch: {len(self.val_loader)}')

        for epoch in range(self.start_epoch, self.total_epoch + 1):
            if self.is_main_process:
                self.logger.info(f'\nstarting training epoch {epoch}')

            epoch_start_time = get_time(self.is_cuda)
            self.run_train_process(epoch)
            val_dice = self.run_val_process(epoch)

            self._save_checkpoint(epoch, val_dice)
            if self.is_main_process:
                self.logger.info(f'end of epoch {epoch}, time: {get_time(self.is_cuda)-epoch_start_time}')

        if self.cfg.model.is_refine:
            self.run_refine()

        if self.is_main_process:
            self.train_writer.close()
            self.val_writer.close()
            self.logger.info(f'\nend of training, best dice: {self.best_dice}')
            run_end_time = get_time(self.is_cuda)
            self.logger.info(f'training time: {(run_end_time-run_start_time) / 60 / 60} hours')

    def run_refine(self):
        train_dataset = semisegdataset(self.cfg, 'refine')
        self.train_loader = dataloaderx(
            dataset=train_dataset,
            batch_size=self.cfg.data_loader.batch_size,
            num_workers=self.num_worker,
            shuffle=true if self.train_sampler is none and not self.cfg.data_loader.semi_dataset else false,
            drop_last=false,
            pin_memory=true,
            sampler=self.train_sampler)
        for epoch in range(self.total_epoch, self.total_epoch + self.cfg.model.refine_epoch):
            if self.is_main_process:
                self.logger.info(f'\nstarting refine epoch {epoch}')

            epoch_start_time = get_time(self.is_cuda)
            self.run_train_process(epoch)
            val_dice = self.run_val_process(epoch)

            self._save_checkpoint(epoch, val_dice)
            if self.is_main_process:
                self.logger.info(f'end of epoch {epoch}, time: {get_time(self.is_cuda) - epoch_start_time}')

    @staticmethod
    def _get_lr(epoch, num_epochs, init_lr):
        if epoch <= num_epochs * 0.66:
            lr = init_lr
        elif epoch <= num_epochs * 0.86:
            lr = init_lr * 0.1
        else:
            lr = init_lr * 0.05

        return lr

    def run_train_process(self, epoch):
        self.model.train()

        train_dice = [0.] * len(self.cfg.data_loader.label_index)
        train_total = 0

        current_lr = none
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._get_lr(epoch, self.total_epoch, self.lr)
            if current_lr is none:
                current_lr = param_group['lr']

        for index, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()

            output_seg = self.model(images)
            seg_loss = self.train_loss_func(output_seg, masks)

            if self.is_apex_train:
                with amp.scale_loss(seg_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                seg_loss.backward()

            # torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10, norm_type=2)

            self.optimizer.step()
            # self.lr_scheduler.step()
            # current_lr = self.optimizer.param_groups[0]["lr"]
            dice_output = self.train_metric_func(output_seg, masks)

            if self.is_distributed_train:
                dice_output = reduce_tensor(dice_output.data)
                seg_loss = reduce_tensor(seg_loss.data)
            for i, dice_tmp in enumerate(dice_output):
                train_dice[i] += float(dice_tmp.item())
            train_total += len(images)

            if self.is_main_process:
                if index > 0 and index % self.cfg.slover.save_frequency == 0:
                    self.logger.info('epoch: {}/{} [{}/{} ({:.0f}%)]'.format(
                        epoch, self.total_epoch,
                        index * len(images), len(self.train_loader),
                        100. * index / len(self.train_loader)))
                    self.logger.info('segloss:{:.6f}, learnrate:{:.6f}'.format(seg_loss.item(), current_lr))

                    for i, dice_label in enumerate(train_dice):
                        dice_ind = dice_label / train_total
                        self.logger.info('{} dice:{:.6f}'.format(self.cfg.data_loader.label_name[i], dice_ind))

            if self.cfg.model.is_dynamic_empty_cache:
                del images, masks, output_seg
                torch.cuda.empty_cache()

        if self.is_main_process:
            self.train_writer.add_scalar('train/segloss', seg_loss.item(), epoch)
            self.train_writer.add_scalar('train/learnrate', current_lr, epoch)

            for i, dice_label in enumerate(train_dice):
                dice_ind = dice_label / train_total
                self.train_writer.add_scalars('train/dice',
                                              {self.cfg.data_loader.label_name[i]: dice_ind}, epoch)

    def run_val_process(self, epoch):
        self.model.eval()

        val_dice = [0.] * len(self.cfg.data_loader.label_index)
        val_total = 0
        val_loss = 0

        for index, (images, masks) in enumerate(self.val_loader):
            images, masks = images.to(self.device), masks.to(self.device)

            with torch.no_grad():
                output_seg = self.model(images)

            seg_loss = self.val_loss_func(output_seg, masks)
            dice_output = self.val_metric_func(output_seg, masks)

            if self.is_distributed_train:
                seg_loss = reduce_tensor(seg_loss.data)
                dice_output = reduce_tensor(dice_output.data)
            val_loss += float(seg_loss.item())
            for i, dice_tmp in enumerate(dice_output):
                val_dice[i] += float(dice_tmp.item())
            val_total += len(images)

            if self.cfg.model.is_dynamic_empty_cache:
                del images, masks, output_seg
                torch.cuda.empty_cache()

        val_loss /= val_total
        total_dice = 0
        if self.is_main_process:
            self.logger.info('loss of validation is {}'.format(val_loss))
            self.val_writer.add_scalar('val/loss', val_loss, epoch)

            for idx, _ in enumerate(val_dice):
                val_dice[idx] /= val_total
                self.logger.info('{} dice:{:.6f}'.format(self.cfg.data_loader.label_name[idx], val_dice[idx]))
                self.val_writer.add_scalars('val/dice',
                                            {self.cfg.data_loader.label_name[idx]: val_dice[idx]}, epoch)
                total_dice += val_dice[idx]
            total_dice /= len(val_dice)
            self.logger.info(f'average dice: {total_dice}')

        return total_dice

    def _load_checkpoint(self):
        if self.is_main_process and self.pretrain_model_path is not none \
                and os.path.exists(self.pretrain_model_path):
            checkpoint = torch.load(self.pretrain_model_path)
            # self.fine_model.load_state_dict(
            #     {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
            model_dict = self.model.state_dict()
            pretrained_dict = checkpoint['state_dict']

            # filter out unnecessary keys
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()
                               if k.replace('module.', '') in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.model.load_state_dict(model_dict)
            self.logger.info(f'load model weight success!')

    def _save_checkpoint(self, epoch, dice):
        if self.is_main_process and dice > self.best_dice:
            self.best_dice = dice
            # "lr_scheduler_dict": self.lr_scheduler.state_dict()},
            torch.save({
                'lr': self.lr,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_dict': self.optimizer.state_dict()},
                self.save_weight_path)
        if self.is_main_process and self.cfg.slover.save_checkpoints and epoch in \
                [int(self.total_epoch*0.36), int(self.total_epoch*0.66), int(self.total_epoch*0.86), self.total_epoch]:
            save_weight_path = os.path.join(self.model_dir, f'epoch_{str(epoch)}_model.pt')
            torch.save({
                'lr': self.lr,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_dict': self.optimizer.state_dict()},
                save_weight_path)
