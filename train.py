import torch
import os
import numpy as np
import random
import sys
from tqdm import tqdm
from model import model, datasets
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    torch.manual_seed(1111)
    np.random.seed(1111)
    random.seed(1111)
    if device == 'cuda':
        torch.cuda.manual_seed(1111)

    trainset = datasets.VOCDataset('train',
                                   'frcnn/data/VOCdevkit/VOC2012/JPEGImages',
                                   'frcnn/data/VOCdevkit/VOC2012/Annotations')
    train_data = DataLoader(trainset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=4)

    frcnn_model = model.FasterRCNN(num_classes=21)
    frcnn_model.train()
    frcnn_model.to(device)

    optimizer = torch.optim.SGD(lr=0.001,
                                params=filter(lambda p: p.requires_grad,
                                              frcnn_model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[12, 16], gamma=0.1)

    acc_steps = 1
    num_epochs = 5
    step_count = 1

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        optimizer.zero_grad()

        for im, target, fname in tqdm(train_data):
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            rpn_output, frcnn_output = frcnn_model(im, target)

            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
            loss = rpn_loss + frcnn_loss

            rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())
            loss = loss / acc_steps
            loss.backward()
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1
        print('Finished epoch {}'.format(i))
        optimizer.step()
        optimizer.zero_grad()
        torch.save(frcnn_model.state_dict(), os.path.join('frcnn', 'faster_rcnn_voc.pth'))
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)
        scheduler.step()
    print('Done Training...')


if __name__ == '__main__':
    train()
