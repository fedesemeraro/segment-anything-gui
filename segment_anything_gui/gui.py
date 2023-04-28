import numpy as np
from matplotlib import pyplot as plt
import cv2
import matplotlib as mpl
from segment_anything import sam_model_registry, SamPredictor
import torch
import os
from matplotlib.patches import Circle


def run_gui(img_input_filepath):
    img = cv2.imread(img_input_filepath, cv2.IMREAD_UNCHANGED)
    
    if img is None: 
        raise Exception("Image file not found.")
    
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    segmenter = Segmenter(img)
    plt.show(block=True)
    return segmenter


class Segmenter():
    def __init__(self, img):
        self.img = img
        self.min_mask_region_area = 500

        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        if torch.cuda.is_available():
            self.sam.to(device="cuda")
        self.predictor = SamPredictor(self.sam)
        
        print("Creating image embeddings ... ", end="")
        self.predictor.set_image(self.img)
        print("Done")
        
        self.color_set = set()
        self.current_color = self.pick_color()
        self.add_xs, self.add_ys, self.rem_xs, self.rem_ys, self.trace = [], [], [], [], []
        
        self.fig, self.ax = plt.subplots(figsize=(10 * (self.img.shape[1] / max(self.img.shape)), 
                                                  10 * (self.img.shape[0] / max(self.img.shape))))
        self.fig.suptitle(f'Segment Anything GUI', fontsize=16)
        self.ax.set_title("Press 'h' to show/hide commands.", fontsize=10)
        self.im = self.ax.imshow(self.img, cmap=mpl.cm.gray)
        self.ax.autoscale(False)
        self.label = 1
        self.add_plot, = self.ax.plot([], [], 'o', markerfacecolor='green', markeredgecolor='black', markersize=5)
        self.rem_plot, = self.ax.plot([], [], 'x', markerfacecolor='red', markeredgecolor='red', markersize=5)
        self.mask_data = np.zeros((self.img.shape[0], self.img.shape[1], 4), dtype=np.uint8)
        for i in range(3):
            self.mask_data[:, :, i] = self.current_color[i]
        self.mask_plot = self.ax.imshow(self.mask_data)
        self.prev_mask_data = np.zeros((self.img.shape[0], self.img.shape[1], 4), dtype=np.uint8)
        self.prev_mask_plot = self.ax.imshow(self.prev_mask_data)
        self.contour_plot, = self.ax.plot([], [], color='black')
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        self.show_help_text = False
        self.help_text = plt.text(0.5, 0.5, '', horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes)
        self.opacity = 120  # out of 255
        self.global_masks = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)
        self.last_mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=bool)  # to undo
        self.full_legend = []
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
    def save_annotation(self, labels_file_outpath):
        dir_path = os.path.split(labels_file_outpath)[0]
        if dir_path != '' and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        cv2.imwrite(labels_file_outpath, self.global_masks)
        print(f"Saved annotation at {labels_file_outpath}")
    
    def pick_color(self):
        while True:
            color = tuple(np.random.randint(low=0, high=255, size=3).tolist())
            if color not in self.color_set:
                self.color_set.add(color)
                return color

    def _on_key(self, event):
        if event.key == 'z':
            self.undo()

        elif event.key == 'enter':
            self.new_tow()

        elif event.key == 'escape':  # save for notebooks
            plt.close(self.fig)
            
        elif event.key == 'h':
            if not self.show_help_text:
                self.help_text.set_text('• \'left click\': select a point inside object to label\n'
                                   '• \'right click\': select a point to exclude from label\n'
                                   '• \'enter\': confirm current label and create new\n'
                                   '• \'z\': undo point\n'
                                   '• \'esc\': close and save')
                self.help_text.set_bbox(dict(facecolor='white', alpha=1, edgecolor='black'))
                self.show_help_text = True
            else:
                self.help_text.set_text('')
                self.show_help_text = False
            self.fig.canvas.draw()
        
    def _on_click(self, event):
        if event.inaxes != self.ax and (event.button in [1, 3]): return
        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))

        if event.button == 1: # left click
            self.trace.append(True)
            self.add_xs.append(x)
            self.add_ys.append(y)
            self.show_points(self.add_plot, self.add_xs, self.add_ys)
            
        else: # right click
            self.trace.append(False)
            self.rem_xs.append(x)
            self.rem_ys.append(y)
            self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)
        
        self.get_mask()
        
    def show_points(self, plot, xs, ys):
        plot.set_data(xs, ys)
        self.fig.canvas.draw()

    def clear_mask(self):
        self.contour_plot.set_data([], [])
        self.mask_data.fill(0)
        self.mask_plot.set_data(self.mask_data)
        self.fig.canvas.draw()

    def get_mask(self):
        mask, _, _ = self.predictor.predict(point_coords=np.array(list(zip(self.add_xs, self.add_ys)) + 
                                                                  list(zip(self.rem_xs, self.rem_ys))), 
                                            point_labels=np.array([1] * len(self.add_xs) + [0] * len(self.rem_xs)), 
                                            multimask_output=False)
        mask = mask[0].astype(np.uint8)
        
        mask[self.global_masks > 0] = 0
        mask = self.remove_small_regions(mask, self.min_mask_region_area, "holes")
        mask = self.remove_small_regions(mask, self.min_mask_region_area, "islands")

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        xs, ys = [], []
        for contour in contours:  # nan to disconnect contours
            xs.extend(contour[:, 0, 0].tolist() + [np.nan])
            ys.extend(contour[:, 0, 1].tolist() + [np.nan])
        self.contour_plot.set_data(xs, ys)

        self.mask_data[:, :, 3] =  mask * self.opacity
        self.mask_plot.set_data(self.mask_data)
        self.fig.canvas.draw()
    
    def undo(self):
        if len(self.trace) == 0:  # undo last mask
            self.global_masks[self.last_mask] = 0
            self.prev_mask_data[:, :, 3][self.last_mask] = 0
            self.prev_mask_plot.set_data(self.prev_mask_data)
            self.label -= 1
            self.full_legend.pop()
            self.ax.legend(handles=self.full_legend, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
            self.clear_mask()
            
        else:  # undo last point
            if self.trace[-1]:
                self.add_xs = self.add_xs[:-1]
                self.add_ys = self.add_ys[:-1]
                self.show_points(self.add_plot, self.add_xs, self.add_ys)
            else:
                self.rem_xs = self.rem_xs[:-1]
                self.rem_ys = self.rem_ys[:-1]
                self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)

            self.trace.pop()
            
            if len(self.trace) != 0:
                self.get_mask()
            else:
                self.clear_mask()

    def new_tow(self):
        # clear points
        self.add_xs, self.add_ys, self.rem_xs, self.rem_ys, self.trace = [], [], [], [], []
        self.show_points(self.add_plot, self.add_xs, self.add_ys)
        self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)
        
        mask = self.mask_data[:, :, 3] > 0
        self.global_masks[mask] = self.label
        self.last_mask = mask.copy()

        self.prev_mask_data[:, :, :3][mask] = self.current_color
        self.prev_mask_data[:, :, 3][mask] = 255
        self.prev_mask_plot.set_data(self.prev_mask_data)
        
        self.full_legend.append(Circle(1, color=np.array(self.current_color)/255, label=f'{self.label}'))
        self.ax.legend(handles=self.full_legend, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
        
        self.current_color = self.pick_color()
        self.label += 1
        
        for i in range(3):
            self.mask_data[:, :, i] = self.current_color[i]
        self.clear_mask()

    @staticmethod
    def remove_small_regions(mask, area_thresh, mode):
        """Function from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py"""
        assert mode in ["holes", "islands"]
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
        sizes = stats[:, -1][1:]  # Row 0 is background label
        small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
        if len(small_regions) == 0:
            return mask
        fill_labels = [0] + small_regions
        if not correct_holes:
            fill_labels = [i for i in range(n_labels) if i not in fill_labels]
            # If every region is below threshold, keep largest
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 1]
        mask = np.isin(regions, fill_labels)
        return mask
