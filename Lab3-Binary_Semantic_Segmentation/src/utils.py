
import numpy as np
def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    pred_mask = np.round(pred_mask.flatten())
    #print(pred_mask)
    gt_mask = gt_mask.flatten()

    fp = np.sum(pred_mask*gt_mask)
    #print((2*fp)/(np.sum(pred_mask)+np.sum(gt_mask)))
    
    return (2*fp)/(np.sum(pred_mask)+np.sum(gt_mask))
