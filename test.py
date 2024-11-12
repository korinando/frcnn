import torch
from model import model, datasets, utils
import cv2
import numpy as np
import random
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_and_dataset():
    seed = 1311
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc = datasets.VOCDataset('test',
                              'frcnn/data/VOCdevkit/VOC2007/JPEGImages',
                              'frcnn/data/VOCdevkit/VOC2007/Annotations')
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False)

    faster_rcnn_model = model.FasterRCNN(num_classes=21)
    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    faster_rcnn_model.load_state_dict(torch.load(os.path.join('frcnn',
                                                              'faster_rcnn_voc.pth'),
                                                 map_location=device))
    return faster_rcnn_model, voc, test_dataset


def infer():
    if not os.path.exists('frcnn/samples'):
        os.mkdir('frcnn/samples')
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset()

    faster_rcnn_model.roi_head.low_score_threshold = 0.7

    for sample_count in tqdm(range(10)):
        random_idx = random.randint(0, len(voc))
        im, target, fname = voc[random_idx]
        im = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()

        # Saving images with ground truth boxes
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            text = voc.idx[target['labels'][idx].detach().cpu().item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=voc.idx[target['labels'][idx].detach().cpu().item()],
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite('frcnn/samples/output_frcnn_gt_{}.png'.format(sample_count), gt_im)

        # Getting predictions from trained model
        rpn_output, frcnn_output = faster_rcnn_model(im, None)
        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        im = cv2.imread(fname)
        im_copy = im.copy()

        # Saving images with predicted boxes
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text = '{} : {:.2f}'.format(voc.idx[labels[idx].detach().cpu().item()],
                                        scores[idx].detach().cpu().item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(im, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        cv2.imwrite('frcnn/samples/output_frcnn_{}.jpg'.format(sample_count), im)


def evaluate_map():
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset()
    gts = []
    preds = []
    for im, target, _ in tqdm(test_dataset):
        im = im.float().to(device)
        target_boxes = target['bboxes'].float().to(device)[0]
        target_labels = target['labels'].long().to(device)[0]
        rpn_output, frcnn_output = faster_rcnn_model(im, None)

        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']

        pred_boxes = {}
        gt_boxes = {}
        for label_name in voc.label:
            pred_boxes[label_name] = []
            gt_boxes[label_name] = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = labels[idx].detach().cpu().item()
            score = scores[idx].detach().cpu().item()
            label_name = voc.idx[label]
            pred_boxes[label_name].append([x1, y1, x2, y2, score])
        for idx, box in enumerate(target_boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = target_labels[idx].detach().cpu().item()
            label_name = voc.idx[label]
            gt_boxes[label_name].append([x1, y1, x2, y2])

        gts.append(gt_boxes)
        preds.append(pred_boxes)

    mean_ap, all_aps = utils.mean_average_precision(preds, gts, method='interp')
    print('Class Wise Average Precisions')
    for idx in range(len(voc.idx)):
        print('AP for class {} = {:.4f}'.format(voc.idx[idx], all_aps[voc.idx[idx]]))
    print('Mean Average Precision : {:.4f}'.format(mean_ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for Faster-RCNN testing')
    parser.add_argument('--evaluate', dest='evaluate',
                        default=False, type=bool)
    parser.add_argument('--infer', dest='infer_samples',
                        default=True, type=bool)
    args = parser.parse_args()

    if args.infer_samples:
        infer()
    else:
        print('Not Inferring for samples')

    if args.evaluate:
        evaluate_map()
    else:
        print('Not Evaluating')