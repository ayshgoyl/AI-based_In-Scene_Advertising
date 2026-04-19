import json
import numpy as np
import pycocotools.mask as maskUtils
import cv2

def evaluate_metrics(gt_path, pred_path):
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)

    # Dictionary mapping image_id to GT RLEs
    gt_masks = {}
    for ann in gt_data['annotations']:
        img_id = ann['image_id']
        if img_id not in gt_masks:
            gt_masks[img_id] = []
        seg = ann['segmentation']
        
        # In COCO, string counts is encoded RLE. If it's a list, it's polygon.
        if isinstance(seg, dict) and 'counts' in seg:
            if isinstance(seg['counts'], str):
                seg['counts'] = seg['counts'].encode('utf-8')
            gt_masks[img_id].append(seg)
        elif isinstance(seg, list):
            rle = maskUtils.frPyObjects(seg, gt_data['images'][img_id]['height'], gt_data['images'][img_id]['width'])
            gt_masks[img_id].append(rle[0])

    height = pred_data['height']
    width = pred_data['width']
    
    ious = []
    accuracies = []
    
    for i, pred in enumerate(pred_data['predictions']):
        frame_idx = pred['frame_idx']
        # Very simple matching: assume image_id perfectly matches frame_idx, or just sequentially if it's a subset
        img_id = list(gt_masks.keys())[i] if i < len(gt_masks) else None
        if img_id is None:
            break
            
        gt_rles = gt_masks[img_id]
        
        # Pred mask
        poly = np.array(pred['polygon'], dtype=np.int32)
        flat_poly = poly.flatten().tolist()
        pred_rle = maskUtils.frPyObjects([flat_poly], height, width)[0]
        
        # Calculate IoU (Intersection over Union)
        # iou takes (dt, gt, iscrowd)
        iou_matrix = maskUtils.iou([pred_rle], gt_rles, [0]*len(gt_rles))
        max_iou = float(np.max(iou_matrix)) if iou_matrix.size > 0 else 0.0
        ious.append(max_iou)
        
        # Accuracy = TP+TN / (P+N). 
        # For simplicity, convert to binary arrays
        pred_bin = maskUtils.decode(pred_rle)
        gt_bin = np.zeros((height, width), dtype=np.uint8)
        for rle in gt_rles:
            gt_bin = np.maximum(gt_bin, maskUtils.decode(rle))
            
        correct_pixels = np.sum(pred_bin == gt_bin)
        total_pixels = height * width
        acc = float(correct_pixels) / total_pixels
        accuracies.append(acc)

    return {
        "mIOU": float(np.mean(ious)) if ious else 0.0,
        "PixelAccuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "mAP": float(np.mean([1 if iou > 0.5 else 0 for iou in ious])) if ious else 0.0,
        "total_evaluated_frames": len(ious)
    }

if __name__ == "__main__":
    import sys
    print(evaluate_metrics(sys.argv[1], sys.argv[2]))
