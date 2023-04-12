import pandas as pd
import os
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from tqdm.auto import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
from .collator import DataCollator
import gc

def prepare_dataset(path):
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=42)
    df['patent_text'] = df['patent_text'].apply(lambda x: 'Generate USPC labels for the following text: ' + x)
    return df

def retrieve_tokenizer(t5_model_name):
    return T5TokenizerFast.from_pretrained(t5_model_name)

def retrieve_model(t5_model_name):
    return T5ForConditionalGeneration.from_pretrained(t5_model_name)

def retrieve_optimizer(t5_model_name):
    model = retrieve_model(t5_model_name)
    return AdamW(model.parameters(), lr=1e-3, eps= 1e-8)

def prepare_data(df, tokenizer):
    gc.collect()
    train_args = {
        'tokenizer': tokenizer,
        'input': df['patent_text'],
        'targets': df['subclass_id'],
    }

    train = DataCollator(**train_args)
    train(n_batches=50)

    train_loader = DataLoader(
        train.datasets, 
        # collate_fn = DataCollatorForSeq2Seq(
        #     tokenizer = tokenizer,
        #     model = model,
        #     pad_to_multiple_of = 8,
        # ),
        batch_size = 4, 
        shuffle=True
    )

    return train_loader

def configure_settings(t5_model_name, epochs, path):
    df = prepare_data(path)
    tokenizer = retrieve_tokenizer(t5_model_name)
    model = retrieve_model(t5_model_name)
    train_loader = prepare_data(df, tokenizer)
    optimizer = retrieve_optimizer(t5_model_name)
    total_steps = len(train_loader) * epochs
    num_warmups = round(total_steps * .05)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmups, num_training_steps=total_steps)

    return {
        'model': model,
        'train_loader': train_loader,
        'optimizer': optimizer,
        'scheduler': scheduler
    }

def trainer(t5_model_name, epochs, path):
    settings = configure_settings(t5_model_name, epochs, path)
    losses = []
    print_freq = 200
    accelerator = Accelerator()

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        settings['model'], settings['optimizer'], settings['train_loader'], settings['scheduler']
    )

    model.train()
    model.to(accelerator.device)
    for epoch_idx in tqdm(range(1, epochs+1)):
        progress_bar = tqdm(train_loader, desc='Epoch {:1d}'.format(epoch_idx), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()
            optimizer.zero_grad()
            batch = tuple(b.to(accelerator.device) for b in batch)
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'labels':         batch[2],
                }       

            outputs = model(**inputs)
        
            loss = outputs.loss
            losses.append(loss.item())
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), os.path.join(os.getcwd(), f'model_checkpoints/Amira-{epoch_idx}.pt'))
        # torch.save(model.state_dict(), '/content/drive/MyDrive/Marvin-xla-large-model.pt')
        # torch.save(optimizer.state_dict(), '/content/drive/MyDrive/Marvin-clalarge-optim.pt')

        avg_loss = np.mean(losses[-print_freq:])
        tqdm.write(f'\nEpoch {epoch_idx}')
        tqdm.write(f'\nAvg. loss: {avg_loss}')
        tqdm.write(f'\nLearning rate: {scheduler.get_last_lr()[0]}')

if __name__ == '__main__':

    T5_MODEL_NAME = 'google/flan-t5-small'
    EPOCHS = 25
    PATH = os.path.join(os.getcwd(), 'data/grouped_labels.csv')

    trainer(
        T5_MODEL_NAME,
        EPOCHS,
        PATH
    )

