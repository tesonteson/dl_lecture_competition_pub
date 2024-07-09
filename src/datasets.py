from src.preprocs import process_text

from statistics import mode
import os
import torch
from PIL import Image
import pandas
from transformers import AutoTokenizer



def collate_fn(batch):
    images, input_ids, attention_masks, answers, mode_answers = zip(*batch)
    images = torch.stack(images)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    answers = torch.stack(answers)
    mode_answers = torch.tensor(mode_answers)
    return images, input_ids, attention_masks, answers, mode_answers


def collate_fn_test(batch):
    images, input_ids, attention_masks = zip(*batch)
    images = torch.stack(images)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    return images, input_ids, attention_masks


class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, tokenizer_path='bert-base-uncased'):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pandas.read_json(df_path)
        self.answer = answer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}

        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question = self.tokenizer(
            process_text(self.df["question"][idx]),
            return_tensors='pt',
            padding='max_length',
            max_length=50,
            truncation=True
        )
        question_input_ids = question['input_ids'].squeeze(0)
        question_attention_mask = question['attention_mask'].squeeze(0)

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)
            return image, question_input_ids, question_attention_mask, torch.Tensor(answers), int(mode_answer_idx)
        else:
            return image, question_input_ids, question_attention_mask

    def __len__(self):
        return len(self.df)
