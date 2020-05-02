import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from utils.dice_loss import dice_coeff


def eval_net(net, loader, device, running_metrics_val, best_iou, writer, logging, epoch):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0.0
    cur_iou = best_iou

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                pred = mask_pred.data.max(1)[1].cpu().numpy()
                gt = true_masks.data.cpu().numpy()
                running_metrics_val.update(gt, pred)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks.squeeze(1), ignore_index=250).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
        logging.info("{}: {}".format(k, v))
        writer.add_scalar("val_metrics/{}".format(k), v, i)

    for k, v in class_iou.items():
        print(k, v)
        logging.info("{}: {}".format(k, v))
        writer.add_scalar("val_metrics/cls_{}".format(k), v, i)

    running_metrics_val.reset()

    if score["Mean IoU : \t"] >= best_iou:
        cur_iou = score["Mean IoU : \t"]
        state = {
            "epoch": epoch + 1,
            "model_state": net.state_dict(),
            "best_iou": cur_iou,
        }
        torch.save(state,
                   'best_iou/' + f'best_model_epoch{epoch + 1}.pth')

    return tot / n_val, cur_iou

def test_net(net, loader, device, running_metrics_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_test = len(loader)  # the number of batch
    tot = 0.0

    with tqdm(total=n_test, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                pred = mask_pred.data.max(1)[1].cpu().numpy()
                gt = true_masks.data.cpu().numpy()
                running_metrics_val.update(gt, pred)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks.squeeze(1), ignore_index=250).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)

    for k, v in class_iou.items():
        print(k, v)

    print(tot / n_test)