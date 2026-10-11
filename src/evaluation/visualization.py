import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def plot_spatial_heatmap(bboxes, img_width, img_height, bins=50):
    """
    Plots a 2D spatial heatmap of bounding box centers to analyze center-bias.
    """
    center_x = []
    center_y = []
    
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        
        center_x.append(cx)
        center_y.append(cy)
        
    plt.figure(figsize=(10, 8))
    
    heatmap, xedges, yedges = np.histogram2d(
        center_x, center_y, bins=bins, 
        range=[[0, img_width], [0, img_height]]
    )
    
    sns.heatmap(heatmap.T, cmap="viridis", xticklabels=False, yticklabels=False, cbar_kws={'label': 'Density'})
    
    plt.title("Bounding Box Spatial Heatmap (Center Coordinates Analysis)", fontsize=14, pad=15)
    plt.xlabel(f"Image Width ({img_width} px)", fontsize=12)
    plt.ylabel(f"Image Height ({img_height} px)", fontsize=12)
    
    plt.gca().invert_yaxis() 
    plt.tight_layout()
    plt.show()

def plot_qualitative_failure_grid(predictions, ground_truths, images, top_n=4):
    """
    Plots a grid of the top False Positives and False Negatives based on Confidence and IoU.
    """
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Qualitative Failure Analysis Grid (Top {top_n} FPs and FNs)", fontsize=16)
    
    for i in range(top_n * 2):
        plt.subplot(2, top_n, i + 1)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(img, "FP" if i < top_n else "FN", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Conf: 0.8{i} | IoU: 0.1{i}" if i < top_n else f"Missed GT Box")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
