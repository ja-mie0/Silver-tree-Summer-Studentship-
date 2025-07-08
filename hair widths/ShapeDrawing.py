import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import pickle
import os
from EdgeDetection import get_hair_contours
from matplotlib.widgets import Button, Slider

def measure_width_along_spine_from_edges(edges, contour):
    # Create a filled mask from contour
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = mask // 255

    # Skeletonize the mask
    skeleton = skeletonize(mask).astype(np.uint8)

    coords = np.column_stack(np.where(skeleton > 0))
    width_map = []

    # Precompute gradients
    gy, gx = np.gradient(skeleton.astype(np.float32))

    for y, x in coords[::max(1, len(coords)//20)]:  # Sample max 20 points
        dx = -gy[y, x]
        dy = gx[y, x]
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue
        dx /= norm
        dy /= norm

        profile = []
        for offset in np.linspace(-15, 15, num=30):  # 30 pixel profile
            px = int(round(x + dx * offset))
            py = int(round(y + dy * offset))
            if 0 <= px < edges.shape[1] and 0 <= py < edges.shape[0]:
                profile.append(mask[py, px])
            else:
                profile.append(0)

        on_line = np.where(np.array(profile) > 0)[0]
        if len(on_line) >= 2:
            width = on_line[-1] - on_line[0]
            width_map.append((x, y, width))
    return width_map, skeleton


def choose_canny_thresholds_with_button(image_path, initial_lower=50, initial_upper=150):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image.")
        return initial_lower, initial_upper

    # Initial values
    lower = initial_lower
    upper = initial_upper

    # Matplotlib UI
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.35)  # Leave more space for widgets

    canny_img = cv2.Canny(img, lower, upper)
    im_disp = ax.imshow(np.hstack([img, canny_img]), cmap='gray')
    ax.set_title('Adjust sliders, then click Accept or press Enter')
    ax.axis('off')

    # Sliders
    ax_lower = plt.axes([0.15, 0.20, 0.7, 0.03])
    ax_upper = plt.axes([0.15, 0.15, 0.7, 0.03])
    slider_lower = Slider(ax_lower, 'Lower', 0, 500, valinit=lower, valstep=1)
    slider_upper = Slider(ax_upper, 'Upper', 0, 500, valinit=upper, valstep=1)

    # Button (bottom center)
    ax_accept = plt.axes([0.4, 0.05, 0.2, 0.06])
    btn_accept = Button(ax_accept, 'Accept', color='lightgreen', hovercolor='yellow')

    result = {'done': False}

    def update(val=None):
        l = int(slider_lower.val)
        u = int(slider_upper.val)
        canny_img = cv2.Canny(img, l, u)
        im_disp.set_data(np.hstack([img, canny_img]))
        fig.canvas.draw_idle()

    slider_lower.on_changed(update)
    slider_upper.on_changed(update)

    def accept(event=None):
        result['done'] = True
        plt.close(fig)

    btn_accept.on_clicked(accept)

    # Keyboard handler
    def on_key(event):
        if event.key == 'enter':
            accept()

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    return int(slider_lower.val), int(slider_upper.val)


def review_and_save_hairs(image_path, output_data_path="hair_measurements.pkl", dataset_path="all_hairs.pkl", scale_microns_per_pixel=1.0):
    # Load image and grayscale for overlays
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lower, upper = choose_canny_thresholds_with_button(image_path)
    contours = get_hair_contours(image_path, lower, upper)
    all_hairs = []
    if os.path.exists(dataset_path):
        with open(dataset_path, "rb") as f:
            all_hairs = pickle.load(f)
    i = 0
    while i < len(contours):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area <= 100:
            i += 1
            continue
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        skeleton = skeletonize(mask // 255)
        width_map, skeleton = measure_width_along_spine_from_edges(cv2.Canny(gray, lower, upper), cnt)

        # Convert widths to microns
        width_map_microns = [(x, y, w * scale_microns_per_pixel) for x, y, w in width_map]
        # Zoomed-in region
        x, y, w, h = cv2.boundingRect(cnt)
        pad = 20
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(mask.shape[1], x+w+pad), min(mask.shape[0], y+h+pad)
        roi = img[y0:y1, x0:x1].copy()
        roi_mask = mask[y0:y1, x0:x1]
        roi_skel = skeleton[y0:y1, x0:x1]
        # Draw overlays
        overlay = roi.copy()
        skel_pts = np.column_stack(np.where(roi_skel > 0))
        for pt in skel_pts:
            cv2.circle(overlay, (pt[1], pt[0]), 1, (0,0,255), -1)
        for xw, yw, wval in width_map_microns:
            if x0 <= xw < x1 and y0 <= yw < y1:
                cv2.putText(overlay, f"{wval:.1f} µm", (xw-x0, yw-y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
        alpha = 0.5
        vis = cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0)
        # Matplotlib UI with Accept/Reject/Quit buttons
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Hair {i+1} (area={area:.1f} px): Accept / Reject / Quit")
        ax.axis('off')
        # Add scale bar
        bar_len_microns = 50  # You can change this value if you want a different reference
        bar_len_px = int(bar_len_microns / scale_microns_per_pixel)
        bar_y = vis.shape[0] - 15
        bar_x0 = 10
        bar_x1 = bar_x0 + bar_len_px
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color='white', lw=3)
        ax.text((bar_x0+bar_x1)//2, bar_y-5, f'{bar_len_microns} µm', color='white', ha='center', va='bottom', fontsize=10, weight='bold')
        # Add buttons
        ax_accept = plt.axes([0.15, 0.01, 0.2, 0.07])
        ax_reject = plt.axes([0.4, 0.01, 0.2, 0.07])
        ax_quit = plt.axes([0.65, 0.01, 0.2, 0.07])
        btn_accept = Button(ax_accept, 'Accept', color='lightgreen')
        btn_reject = Button(ax_reject, 'Reject', color='salmon')
        btn_quit = Button(ax_quit, 'Quit', color='lightgray')
        result = {'action': None}
        def accept(event):
            result['action'] = 'accept'
            plt.close(fig)
        def reject(event):
            result['action'] = 'reject'
            plt.close(fig)
        def quit_(event):
            result['action'] = 'quit'
            plt.close(fig)
        btn_accept.on_clicked(accept)
        btn_reject.on_clicked(reject)
        btn_quit.on_clicked(quit_)
        plt.show()
        if result['action'] == 'accept':
            hair_data = {
                'contour': cnt,
                'width_map': width_map_microns,
                'bounding_box': (x, y, w, h),
                'image_path': image_path,
                'area': area
            }
            all_hairs.append(hair_data)
            with open(output_data_path, "ab") as f:
                pickle.dump(hair_data, f)
            print(f"Hair {i+1} accepted and saved.")
        elif result['action'] == 'quit':
            break
        else:
            print(f"Hair {i+1} rejected.")
        i += 1
    # Save/update the larger dataset
    with open(dataset_path, "wb") as f:
        pickle.dump(all_hairs, f)
    print(f"Total accepted hairs in dataset: {len(all_hairs)}")

def draw_hair_edges(image_path):
    lower, upper = choose_canny_thresholds_with_button(image_path)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Edge detection
    edges = cv2.Canny(gray, lower, upper)

    # Step 3: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Draw contours or bounding boxes
    annotated_img = img.copy()
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue  # Skip small noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        count += 1

    print(f"[INFO] Detected {count} potential hairs.")
    
    # Step 5: Show result
    cv2.imshow("Detected Hair Edges", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calibrate_scale(image_path):
    img = cv2.imread(image_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title("Click two points on a horizontal scale bar, then close the window. (Only last two clicks are used)")
    pts = plt.ginput(n=-1, timeout=0)  # Unlimited clicks until window close
    plt.close(fig)
    if len(pts) < 2:
        print("Calibration failed: two points not selected.")
        return None
    (x1, y1), (x2, y2) = pts[-2:]
    pixel_dist = abs(x2 - x1)
    print(f"You selected a distance of {pixel_dist:.2f} pixels.")
    # Show zoomed-in region
    x_min, x_max = int(min(x1, x2)), int(max(x1, x2))
    y_c = int((y1 + y2) / 2)
    pad = 30  # pixels
    y0 = max(0, y_c - pad)
    y1z = min(img.shape[0], y_c + pad)
    zoom_img = img[y0:y1z, x_min:x_max].copy()
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.imshow(cv2.cvtColor(zoom_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Zoomed region for calibration (close to continue)")
    ax2.plot([0, x_max - x_min], [int(y1 - y0), int(y2 - y0)], color='yellow', lw=3)
    ax2.axis('off')
    plt.show()
    micron_dist = float(input("Enter the real-world micron distance between these points: "))
    scale = micron_dist / pixel_dist
    print(f"Scale set to {scale:.4f} microns per pixel.")
    return scale
# Example usage
if __name__ == "__main__":
    # Interactive scale calibration
    image_path = "hair widths/data/1.jpg"
    scale = calibrate_scale(image_path)
    if scale is not None:
        review_and_save_hairs(image_path, scale_microns_per_pixel=scale)
    else:
        print("Scale calibration failed. Exiting.")

