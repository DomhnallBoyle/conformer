import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from aiayn.sampler import CustomSampler
from aiayn.scheduler import CustomScheduler
from collator import CustomCollator
from dataset import LibriSpeechDataset
from model import E2E
from utils import decode, list_type, plot_mel_tensorboard

# TODO: 
#  if using gradient accumulation, need to make sure the 8 batches contain samples all of the same length


def main(args) -> None:
    train_dataset = LibriSpeechDataset(path=args.dataset_path, sets=args.train_sets, name='train')
    val_dataset = LibriSpeechDataset(path=args.dataset_path, sets=args.val_sets, name='val')

    print('Train dataset size:', len(train_dataset))
    print('Val dataset size:', len(val_dataset))

    gradient_accumulation = args.gradient_accumulation_steps is not None

    # num_classes includes the blank (+1)
    model = E2E(num_classes=train_dataset.num_classes + 1, group_norm=gradient_accumulation).to(config.device)

    # requires log probs (log softmax), targets with class indices (excl. blank), input and target lengths
    criterion = torch.nn.CTCLoss(blank=config.blank, zero_infinity=True)

    # peak LR in scheduler should be 0.05 / sqrt(d) where d is the conformer encoder dim
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr_initial, betas=config.lr_betas, eps=config.lr_eps, weight_decay=config.l2_regularisation_weight)
    lr_scheduler = CustomScheduler(optimiser=optimiser)
    print('Peak LR:', 0.05 / (config.params['d_encoder'] ** 0.5))

    # order: sampler, dataset[i], collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_sampler=CustomSampler(train_dataset, batch_size=args.batch_size),
        collate_fn=CustomCollator(spec_aug=args.spec_aug, debug=args.debug),
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        batch_size=1,
        shuffle=True,
        collate_fn=CustomCollator(),
        pin_memory=True
    )

    model.train()
    model.zero_grad()

    writer = SummaryWriter()

    num_steps = 0
    num_epochs = 0
    running_loss = 0
    finished_training = False

    while not finished_training:
        for train_data in train_loader:
            audios, mels, transcripts, transcript_targets, input_lengths, target_lengths = train_data  # batch

            mels = mels.to(config.device)
            transcript_targets = transcript_targets.to(config.device)
            input_lengths = input_lengths.to(config.device)
            target_lengths = target_lengths.to(config.device)

            output_log_probs, output_probs = model(mels)

            # TODO: remove this
            # update input lengths based on SpecAugment stretching
            # input_lengths = torch.Tensor([output_log_probs.shape[1]] * args.batch_size).int()

            # CTC loss requires log_probs as [T, B, C] from log_softmax()
            loss = criterion(output_log_probs.permute(1, 0, 2), transcript_targets, input_lengths, target_lengths)
            running_loss += loss.item()
            if gradient_accumulation:
                loss /= args.gradient_accumulation_steps  # same as (loss 1 + loss 2 + loss 3) / args.gradient_accumulation_steps
            loss.backward()  # accumulates the gradients from every forward pass

            if not gradient_accumulation or (gradient_accumulation and (num_steps + 1) % args.gradient_accumulation_steps == 0):
                optimiser.step()  # update weights
                optimiser.zero_grad()  # only zero the gradients after every update
                lr_scheduler.step()  # adjusting lr

            # log
            if (num_steps + 1) % args.log_every == 0:
                running_loss /= args.log_every
                num_updates = (num_steps + 1) // args.gradient_accumulation_steps if gradient_accumulation else (num_steps + 1)

                print(f'Num epochs: {num_epochs}, Num steps: {num_steps + 1}, Num updates: {num_updates}, Loss: {running_loss}, LR: {lr_scheduler.lr}')
                
                # output training sample
                gt_transcript = transcripts[0]
                pred_transcript = decode(output_probs[0], decoder=train_dataset.decoder)
                print(f'GT: "{gt_transcript}"\nPred: "{pred_transcript}"\n')

                # tensorboard logging
                for label, value in zip(['Epochs', 'LR', 'Loss/train'], [num_epochs, lr_scheduler.lr, running_loss]):
                    writer.add_scalar(label, value, num_steps + 1)
                writer.add_figure('Mels/train', plot_mel_tensorboard(mels[0]), num_steps + 1)
                writer.add_audio('Audio/train', audios[0], num_steps + 1, sample_rate=config.sample_rate)

                # reset variables
                running_loss = 0

            # run eval
            if (num_steps + 1) % args.val_every == 0:
                print('Running evaluation...')
                val_running_loss = 0

                for i, val_data in enumerate(tqdm(val_loader)):
                    if i == args.val_steps:
                        break

                    val_audios, val_mels, val_transcripts, val_transcript_targets, val_input_lengths, val_target_lengths = val_data

                    val_mels = val_mels.to(config.device)
                    val_transcript_targets = val_transcript_targets.to(config.device)
                    val_input_lengths = val_input_lengths.to(config.device)
                    val_target_lengths = val_target_lengths.to(config.device)

                    val_output_log_probs, val_output_probs = model(val_mels)
                    
                    # CTC loss requires log_probs as [T, B, C] from log_softmax()
                    val_loss = criterion(val_output_log_probs.permute(1, 0, 2), val_transcript_targets, val_input_lengths, val_target_lengths)

                    val_running_loss += val_loss.item()

                val_running_loss /= args.val_steps

                # output val sample, ensure using the training decoder
                val_gt_transcript = val_transcripts[0]
                val_pred_transcript = decode(val_output_probs[0], decoder=train_dataset.decoder)
                print(f'Val Loss: {val_running_loss}')
                print(f'Val GT: "{val_gt_transcript}"\nVal Pred: "{val_pred_transcript}"\n')

                # tensorboard logging
                writer.add_scalar('Loss/val', val_running_loss, num_steps + 1)
                writer.add_figure('Mels/val', plot_mel_tensorboard(val_mels[0]), num_steps + 1)
                writer.add_text('Text/val', f'{val_gt_transcript} | -> | {val_pred_transcript}', num_steps + 1)
                writer.add_audio('Audio/val', val_audios[0], num_steps + 1, sample_rate=config.sample_rate)

            num_steps += 1

        num_epochs += 1

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('train_sets', type=list_type)
    parser.add_argument('val_sets', type=list_type)
    parser.add_argument('batch_size', type=int)  # TODO: data loader seems to hang with large batch sizes?
    parser.add_argument('--num_steps', type=int, default=1_000_000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--val_every', type=int, default=1_000)
    parser.add_argument('--val_steps', type=int, default=50)
    parser.add_argument('--spec_aug', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
