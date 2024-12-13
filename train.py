import argparse

import torch
from tqdm import tqdm

import config
from aiayn.sampler import CustomSampler
from aiayn.scheduler import CustomScheduler
from collator import CustomCollator
from dataset import LibriSpeechDataset
from model import E2E
from utils import list_type, decode


def main(args) -> None:
    train_dataset = LibriSpeechDataset(path=args.dataset_path, sets=args.train_sets, train=True)
    val_dataset = LibriSpeechDataset(path=args.dataset_path, sets=args.val_sets)

    print('Train dataset size:', len(train_dataset))
    print('Val dataset size:', len(val_dataset))

    # num_classes includes the blank (+1)
    model = E2E(num_classes=train_dataset.num_classes + 1).to(config.device)

    # requires log probs (log softmax), targets with class indices (excl. blank), input and target lengths
    criterion = torch.nn.CTCLoss(blank=config.blank, zero_infinity=True)

    # TODO: peak LR in scheduler should be 0.05 / sqrt(d) where d is the conformer encoder dim
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr_initial, betas=config.lr_betas, eps=config.lr_eps, weight_decay=config.l2_regularisation_weight)
    lr_scheduler = CustomScheduler(optimiser=optimiser, d_model=config.params['d_encoder'], warmup_steps=config.warmup_steps)

    collator = CustomCollator()

    # order: sampler, dataset[i], collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_sampler=CustomSampler(train_dataset, batch_size=args.batch_size),
        collate_fn=collator,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True
    )

    model.train()
    model.zero_grad()

    num_steps = 0
    num_epochs = 0
    running_loss = 0
    finished_training = False

    while not finished_training:
        for train_data in train_loader:
            mels, transcript_targets, transcripts, input_lengths, target_lengths = train_data  # batch

            mels = mels.to(config.device)
            transcript_targets = transcript_targets.to(config.device)
            input_lengths = input_lengths.to(config.device)
            target_lengths = target_lengths.to(config.device)

            output_log_probs, output_probs = model(mels, training=True)

            # update input lengths based on SpecAugment stretching
            input_lengths = torch.Tensor([output_log_probs.shape[1]] * args.batch_size).int()

            # CTC loss requires log_probs as [T, B, C] from log_softmax()
            loss = criterion(output_log_probs.permute(1, 0, 2), transcript_targets, input_lengths, target_lengths)
            loss.backward()  # accumulates the gradients from every forward pass

            optimiser.step()  # update weights
            optimiser.zero_grad()  # only zero the gradients after every update

            lr_scheduler.step()  # adjusting lr

            running_loss += loss.item()

            if (num_steps + 1) % args.log_every == 0:
                print(f'Num epochs: {num_epochs}, Num steps: {num_steps}, Loss: {running_loss / args.log_every}, LR: {lr_scheduler.lr}')
                
                # output training sample
                gt_transcript = transcripts[0]
                pred_transcript = decode(output_probs[0], decoder=train_dataset.decoder)
                print(f'GT: "{gt_transcript}"\nPred: "{pred_transcript}"\n')

                # reset variables
                running_loss = 0

            if (num_steps + 1) % args.val_every == 0:
                print('Running evaluation...')
                val_running_loss = 0

                for val_data in tqdm(val_loader):
                    val_mels, val_transcript_targets, val_transcripts, val_input_lengths, val_target_lengths = val_data

                    val_mels = val_mels.to(config.device)
                    val_transcript_targets = val_transcript_targets.to(config.device)
                    val_input_lengths = val_input_lengths.to(config.device)
                    val_target_lengths = val_target_lengths.to(config.device)

                    val_output_log_probs, val_output_probs = model(val_mels, training=False)
                    
                    # CTC loss requires log_probs as [T, B, C] from log_softmax()
                    val_loss = criterion(val_output_log_probs.permute(1, 0, 2), val_transcript_targets, val_input_lengths, val_target_lengths)

                    val_running_loss += val_loss.item()

                val_running_loss /= len(val_loader)

                # output val sample, ensure using the training decoder
                val_gt_transcript = val_transcripts[0]
                val_pred_transcript = decode(output_probs[0], decoder=train_dataset.decoder)
                print(f'Val Loss: {val_running_loss}')
                print(f'Val GT: "{val_gt_transcript}"\nVal Pred: "{val_pred_transcript}"\n')

            num_steps += 1

        num_epochs += 1


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

    main(parser.parse_args())
