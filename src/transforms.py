import numpy as np
from scipy.ndimage import zoom
import copy

def clipped_zoom(img, zoom_factor, channel):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, order = 0, mode = 'constant')

        zm = copy.deepcopy(out) 
        top, bottom, left, right = top, top+zh, left, left+zw
        h_bottom = zm.shape[0] - bottom
        w_right = zm.shape[1] - right
        img_refl = copy.deepcopy(zm)

        if channel == "masks": # Assign different labels if mask is reflected
            RTS_max_label = np.max(np.unique(zm))

            # Apply reflection padding to the top, give new label-------------
            top_part = copy.deepcopy(img_refl[top:top*2, :])
            RTS_label = np.unique(top_part)
            if len(RTS_label)>1: # RTS is in this section -> give it different label
                for label in RTS_label[1:]:
                    if label in top_part[0,]: # Touching lower border, no need to give it a new label
                        next 
                    else: # Assign unused label
                        top_part[top_part == label] = RTS_max_label+1
                        RTS_max_label+=1
                img_refl[:top, :] = np.flipud(top_part)
            else:
                img_refl[:top, :] = np.flipud(top_part)

             # Apply reflection padding at bottom, give new label---------------------   
            bottom_part= copy.deepcopy(img_refl[-h_bottom*2:-h_bottom, :])
            RTS_label = np.unique(bottom_part)
            if len(RTS_label)>1: # RTS is in this section -> give it different label
                for label in RTS_label[1:]:
                    if label in bottom_part[-1,:]: # Touching lower border, no need to give it a new label
                        #print(label)
                        next 
                    else: # Assign unused label
                        bottom_part[bottom_part == label] = RTS_max_label+1
                        RTS_max_label+=1
                img_refl[-h_bottom:, :] = np.flipud(bottom_part)
            else:
                img_refl[-h_bottom:, :] = np.flipud(bottom_part)


            # Apply reflection padding at left side, give new label---------------------   
            left_part= copy.deepcopy(img_refl[:, left:left*2])
            RTS_label = np.unique(left_part)

            if len(RTS_label)>1: # RTS is in this section -> give it different label
                for label in RTS_label[1:]:
                    if label in left_part[:,0]: # Touching left border, no need to give it a new label
                        #print(label)
                        next 

                    else: # Assign unused label
                        left_part[left_part == label] = RTS_max_label+1
                        RTS_max_label+=1

                img_refl[:, :left] = np.fliplr(left_part)
            else:
                img_refl[:, :left] = np.fliplr(left_part)

            # Apply reflection padding at right side, give new label---------------------   
            right_part= copy.deepcopy(img_refl[:, -w_right*2:-w_right])
            RTS_label = np.unique(right_part)
            if len(RTS_label)>1: # RTS is in this section -> give it different label
                for label in RTS_label[1:]:
                    if label in right_part[:,-1]: # Touching left border, no need to give it a new label
                        #print(label)
                        next 
                    else: # Assign unused label
                        right_part[right_part == label] = RTS_max_label+1
                        RTS_max_label+=1
                img_refl[:, -w_right:] = np.fliplr(right_part)
            else:
                img_refl[:, -w_right:] = np.fliplr(right_part)

            output = img_refl
        else:
            # Apply reflection padding to the top and bottom----------------------------------------------------------------
            img_refl[:top, :] = np.flipud(img_refl[top:top*2, :])
            img_refl[-h_bottom:, :] = np.flipud(img_refl[-h_bottom*2:-h_bottom, :])

            # Reflection at left and right
            img_refl[:, :left] = np.fliplr(img_refl[:, left:left*2])
            img_refl[:, -w_right:] = np.fliplr(img_refl[:, -w_right*2:-w_right])
            output = img_refl


        # Zooming in
    elif zoom_factor > 1:
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple,  order = 0, mode = 'constant')

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        output = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        output = img
    return output




