import os
import cv2
import shutil
import hashlib
import numpy as np
import imagehash
from PIL import Image
from sklearn.cluster import KMeans

#Dataset CLeaning
def clean_dataset(base_path, folder_list):
    """Removes corrupt files, exact duplicates (MD5), and near-duplicates (pHash)"""
    for folder in folder_list:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        print(f"\nCleaning started for: {folder}...")

        md5_hashes = set()
        phash_hashes = {}
        deleted_exact = 0
        deleted_near = 0
        deleted_corrupt = 0

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if not os.path.isfile(file_path):
                continue

            # Check for corruption
            try:
                img = Image.open(file_path)
                img.verify()

                img = Image.open(file_path)
                cv_img = cv2.imread(file_path)
                if cv_img is None:
                    raise Exception("OpenCV decoding failed")
            except Exception:
                os.remove(file_path)
                deleted_corrupt += 1
                continue

            # Exact duplicate check (MD5)
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in md5_hashes:
                os.remove(file_path)
                deleted_exact += 1
                continue
            else:
                md5_hashes.add(file_hash)

            # Near duplicate check (pHash)
            try:
                current_phash = imagehash.phashing_dict(imagehash.phash(img))['phash'][0]
                is_near_duplicate = False
                
                for existing_hash in phash_hashes:
                    if imagehash.hex_to_hash(existing_hash) - imagehash.hex_to_hash(current_phash) < 4:
                        os.remove(file_path)
                        deleted_near += 1
                        is_near_duplicate = True
                        break

                if not is_near_duplicate:
                    phash_hashes[current_phash] = file_path
            except Exception:
                pass

        print(f"[{folder}] Corrupt removed: {deleted_corrupt}")
        print(f"[{folder}] Exact duplicates removed: {deleted_exact}")
        print(f"[{folder}] Near duplicates removed: {deleted_near}")
        print(f"[{folder}] Remaining images: {len(os.listdir(folder_path))}")


#Separation
def process_and_grade_dataset(base_raw_path, final_dataset_path):
    """Segments infected leaves and clusters them into 4 pseudo-labeled severity grades"""
    infected_folder = os.path.join(base_raw_path, '3_Infected_Raw')
    healthy_folder = os.path.join(base_raw_path, '1_Healthy')
    dry_folder = os.path.join(base_raw_path, '2_Natural_Dry')

    # Configs for HSV filtering and percentile analytics
    DRY_TRASH_LOWER = np.array([10, 35, 40])
    DRY_TRASH_UPPER = np.array([26, 255, 255])
    INTENSITY_PERCENTILE = 92

    classes = ['Class_0_Healthy', 'Class_1_Dry_Leaves', 'Class_2_Grade_1', 'Class_3_Grade_2', 'Class_4_Grade_3', 'Class_5_Grade_4']
    for cls in classes:
        os.makedirs(os.path.join(final_dataset_path, cls), exist_ok=True)

    print("\nStarting advanced feature extraction using HSV suppression and CLAHE alignment...")
    features = []
    infected_images_list = []

    for filename in os.listdir(infected_folder):
        img_path = os.path.join(infected_folder, filename)
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Scale to 224x224 framework standard
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Eliminate background straw/trash via HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        dry_trash_mask = cv2.inRange(img_hsv, DRY_TRASH_LOWER, DRY_TRASH_UPPER)

        # Contrast enhancement via CLAHE
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl_l = clahe.apply(l_ch)
        img_lab_enhanced = cv2.merge((cl_l, a_ch, b_ch))

        # Background partitioning via local K-means
        pixel_values = np.float32(img_lab_enhanced.reshape((-1, 3)))
        local_kmeans = KMeans(n_clusters=3, random_state=42, n_init=5)
        labels = local_kmeans.fit_predict(pixel_values)
        segmented_labels = labels.reshape(img_resized.shape[:2])
        local_centroids = local_kmeans.cluster_centers_

        # Suppress the background cluster matching trash mask
        trash_counts = {i: np.sum((segmented_labels == i) & (dry_trash_mask == 255)) for i in range(3)}
        dry_trash_cluster = max(trash_counts, key=trash_counts.get)
        remaining_clusters = [i for i in range(3) if i != dry_trash_cluster]

        # Isolate leaf and disease tissue nodes using a* redness channel
        if len(remaining_clusters) == 2:
            c1, c2 = remaining_clusters[0], remaining_clusters[1]
            if local_centroids[c1][1] > local_centroids[c2][1]:
                disease_cluster = c1
                leaf_cluster = c2
            else:
                disease_cluster = c2
                leaf_cluster = c1
        else:
            leaf_cluster = remaining_clusters[0]
            disease_cluster = remaining_clusters[0]

        # Build final mask with morphology and contour filtering
        final_mask = np.zeros_like(segmented_labels, dtype=np.uint8)
        final_mask[(segmented_labels == leaf_cluster) | (segmented_labels == disease_cluster)] = 255
        final_mask[dry_trash_mask == 255] = 0

        kernel = np.ones((5, 5), np.uint8)
        clean_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.zeros_like(clean_mask)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Extract target pixels inside the validated leaf contour
        if np.sum(contour_mask == 255) > (224 * 224 * 0.01):
            a_pixels = img_lab_enhanced[:, :, 1][contour_mask == 255]
            L_pixels = img_lab_enhanced[:, :, 0][contour_mask == 255]
        else:
            a_pixels = img_lab_enhanced[:, :, 1].flatten()
            L_pixels = img_lab_enhanced[:, :, 0].flatten()

        # Calculate high-percentile feature analytics
        top_a_threshold = np.percentile(a_pixels, INTENSITY_PERCENTILE)
        mean_redness = np.mean(a_pixels[a_pixels >= top_a_threshold])

        top_L_threshold = np.percentile(L_pixels, INTENSITY_PERCENTILE)
        mean_lightness = np.mean(L_pixels[L_pixels >= top_L_threshold])

        features.append([mean_lightness, mean_redness])
        infected_images_list.append(img_path)

    X = np.array(features)
    print(f"\nFeature extraction complete for {len(X)} infected samples.")
    print("Running Global K-means Clustering (K=4) for pseudo-label allocation...")

    # Global clustering into 4 severity profiles
    global_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    global_labels = global_kmeans.fit_predict(X)
    global_centroids = global_kmeans.cluster_centers_

    # Sort clusters dynamically: Lightness + 1.5 * Redness
    score_centroids = global_centroids[:, 0] + (1.5 * global_centroids[:, 1])
    sorted_cluster_indices = np.argsort(score_centroids)

    print("Distributing pseudo-labeled sugarcane images into final structured folders...")
    for img_path, current_label in zip(infected_images_list, global_labels):
        filename = os.path.basename(img_path)
        logical_grade = np.where(sorted_cluster_indices == current_label)[0][0]
        target_folder = os.path.join(final_dataset_path, f'Class_{logical_grade + 2}_Grade_{logical_grade + 1}')
        shutil.copy(img_path, os.path.join(target_folder, filename))

    # Sync baseline control frames (Healthy and Natural Dry)
    print("Syncing baseline control frames to final repository paths...")
    for filename in os.listdir(healthy_folder):
        shutil.copy(os.path.join(healthy_folder, filename), os.path.join(final_dataset_path, 'Class_0_Healthy', filename))

    for filename in os.listdir(dry_folder):
        shutil.copy(os.path.join(dry_folder, filename), os.path.join(final_dataset_path, 'Class_1_Dry_Leaves', filename))

    print("\n=== Data Pipeline Execution Terminated Successfully ===")
    for cls in classes:
        print(f"{cls}: {len(os.listdir(os.path.join(final_dataset_path, cls)))} files populated.")


if __name__ == "__main__":
    # Path configuration
    root_dir = '/content/drive/MyDrive/Research2026/Raw_Data/'
    final_output_dir = '/content/drive/MyDrive/Research2026/Sugarcane_Final_Dataset/'
    folders_to_clean = ['1_Healthy', '2_Natural_Dry', '3_Infected_Raw']

    # Step 1: Clean raw dataset from anomalies and duplicates
    clean_dataset(root_dir, folders_to_clean)

    # Step 2: Feature extraction, clustering, and final repository generation
    process_and_grade_dataset(root_dir, final_output_dir)