from src.utils import set_seed
from src.datasets import collate_fn, collate_fn_test, VQADataset
from src.models.base import VQAModel
from train import train

import os
import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms



def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 画像データの前処理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #データセットの作成
    train_dataset = VQADataset(df_path="./data/train.json",
                               answer=True,
                               answer_label_path="./data/class_mapping.csv",
                               image_dir="./data/train",
                               transform=transform,
                               tokenizer_path='bert-base-uncased',
                               create_corpus=True)
    test_dataset = VQADataset(df_path="./data/valid.json",
                              answer=False,
                              answer_label_path="./data/class_mapping.csv",
                              image_dir="./data/valid",
                              transform=transform,
                              tokenizer_path='bert-base-uncased',
                              create_corpus=False)
    test_dataset.update_dict(train_dataset)

    #データローダーの作成
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=os.cpu_count(),
                                               pin_memory=True,
                                               collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=os.cpu_count(),
                                              pin_memory=True,
                                              collate_fn=collate_fn_test)

    #モデルの作成
    model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device)

    #損失関数＆オプティマイザ＆スケジューラの作成
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    #モデルの学習と検証
    num_epoch = 20
    for epoch in tqdm(range(num_epoch)):
        train_loss, train_acc, train_simple_acc, train_time = train(model,
                                                                    train_loader,
                                                                    optimizer,
                                                                    criterion,
                                                                    device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
            f"train time: {train_time:.2f} [s]\n"
            f"train loss: {train_loss:.4f}\n"
            f"train acc: {train_acc:.4f}\n"
            f"train simple acc: {train_simple_acc:.4f}")
        scheduler.step()

    #テストデータ予測
    model.eval()
    submission = []
    for image, question_input_ids, question_attention_mask in tqdm(test_loader):
        image, question_input_ids, question_attention_mask = \
            image.to(device), question_input_ids.to(device), question_attention_mask.to(device)
        pred = model(image, question_input_ids, question_attention_mask)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    #提出ファイル出力
    output_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), f"./src/models/model_{output_date}.pth")
    np.save(f"./data/submission_{output_date}.npy", submission)



if __name__ == "__main__":
    main()
