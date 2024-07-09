import torch
import time



def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.
    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10
    return total_acc / len(batch_pred)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question_input_ids, question_attention_mask, answers, mode_answer in dataloader:
        image, question_input_ids, question_attention_mask, answers, mode_answer = \
            image.to(device), question_input_ids.to(device), question_attention_mask.to(device), answers.to(device), mode_answer.to(device)
        pred = model(image, question_input_ids, question_attention_mask)
        loss = criterion(pred, mode_answer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question_input_ids, question_attention_mask, answers, mode_answer in dataloader:
        image, question_input_ids, question_attention_mask, answers, mode_answer = \
            image.to(device), question_input_ids.to(device), question_attention_mask.to(device), answers.to(device), mode_answer.to(device)
        pred = model(image, question_input_ids, question_attention_mask)
        loss = criterion(pred, mode_answer)
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - star
