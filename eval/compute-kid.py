import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import logging
import argparse
from scipy.linalg import sqrtm

parser = argparse.ArgumentParser()
parser.add_argument("--real_path", type=str, default="/data1/Shared/Data/mvtec_anomaly_detection", help="Path to the real image dataset")
parser.add_argument("--generated_path", type=str, default="./MCA-Ctrl/without_AGO_DAE", help="Path to the generated image dataset")

args = parser.parse_args()

logging.basicConfig(filename=f'{args.generated_path}_kid_score_log.txt',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InceptionV3FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()

    def forward(self, x):
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def get_image_paths_from_folder(folder):
    exts = ['.png', '.jpg', '.jpeg']
    image_paths = []
    if not os.path.isdir(folder):
        return image_paths
    for file in os.listdir(folder):
        if any(file.lower().endswith(ext) for ext in exts):
            img_path = os.path.join(folder, file)
            if os.path.isfile(img_path):
                image_paths.append(img_path)
    return image_paths

def get_activations(image_paths, model, batch_size=32):
    activations = []
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = transform(img)
                    batch_images.append(img)
                except Exception as e:
                    print(f"Image loading error {path}: {e}")

            if batch_images:
                batch_images = torch.stack(batch_images).to(device)
                batch_activations = model(batch_images)
                activations.append(batch_activations.cpu().numpy())

    return np.concatenate(activations, axis=0) if activations else np.array([])

def calculate_kid(activations_real, activations_generated, subsample_size=None, num_subsets=10):
    def polynomial_kernel(x, y):
        d = x.shape[0]
        return (1 + np.dot(x, y) / d) ** 3

    def mmd_squared(x_feats, y_feats):
        n, m = len(x_feats), len(y_feats)
        sum_kxx = sum(polynomial_kernel(x_feats[i], x_feats[j]) for i in range(n) for j in range(i+1, n))
        sum_kyy = sum(polynomial_kernel(y_feats[i], y_feats[j]) for i in range(m) for j in range(i+1, m))
        sum_kxy = sum(polynomial_kernel(x_feats[i], y_feats[j]) for i in range(n) for j in range(m))
        mmd = (2.0 * sum_kxx / (n*(n-1))) + (2.0 * sum_kyy / (m*(m-1))) - (2.0 * sum_kxy / (n*m))
        return mmd

    if activations_real.size == 0 or activations_generated.size == 0:
        return None

    kid_scores = []
    n_real = activations_real.shape[0]
    n_gen = activations_generated.shape[0]

    if subsample_size is None:
        subsample_size = 10 if n_real >= 10 else n_real

    for _ in range(num_subsets):
        idx_real = np.random.choice(n_real, subsample_size, replace=False)
        idx_gen  = np.random.choice(n_gen, subsample_size, replace=False)
        x_feats = activations_real[idx_real]
        y_feats = activations_generated[idx_gen]
        mmd_val = mmd_squared(x_feats, y_feats)
        kid_scores.append(mmd_val)

    return np.mean(kid_scores)

def get_categories_and_classes(folder_path):
    categories = [
        'bottle','cable','capsule','carpet','grid','hazelnut','leather',
        'metal_nut','pill','screw','tile','toothbrush','transistor',
        'wood','zipper'
    ]
    category_class_dict = {}
    for category in categories:
        category_path = os.path.join(folder_path, category)
        if not os.path.isdir(category_path):
            category_class_dict[category] = []
            continue
        defect_classes = [d for d in os.listdir(category_path)
                          if os.path.isdir(os.path.join(category_path, d))]
        category_class_dict[category] = defect_classes
    return category_class_dict

def main():
    real_folder_path = args.real_path
    generated_folder_path = args.generated_path
    category_class_dict = get_categories_and_classes(generated_folder_path)
    model = InceptionV3FeatureExtractor().to(device)
    category_kid_scores = {}

    for category, defect_classes in category_class_dict.items():
        logging.info(f"Processing category: {category}")
        print(f"Processing category: {category}")
        class_kid_scores = []

        for defect_class in defect_classes:
            real_class_folder = os.path.join(real_folder_path, category, 'test', defect_class)
            generated_class_folder = os.path.join(generated_folder_path, category, defect_class)
            real_image_paths = get_image_paths_from_folder(real_class_folder)
            generated_image_paths = get_image_paths_from_folder(generated_class_folder)

            if real_image_paths and generated_image_paths:
                act_real = get_activations(real_image_paths, model)
                act_generated = get_activations(generated_image_paths, model)

                if act_real.size == 0 or act_generated.size == 0:
                    logging.warning(f"No valid activations for {category}/{defect_class}.")
                    print(f"No valid activations for {category}/{defect_class}.")
                    continue

                n_real = act_real.shape[0]
                subsample_size = 10 if n_real >= 10 else n_real
                num_subsets=50

                kid_score = calculate_kid(act_real, act_generated,
                                          subsample_size=subsample_size,
                                          num_subsets=num_subsets)
                if kid_score is not None:
                    class_kid_scores.append(kid_score)
                    logging.info(f"KID score for {category}/{defect_class}: {kid_score * 1000}")
                    print(f"KID score for {category}/{defect_class}: {kid_score * 1000}")
            else:
                logging.warning(f"No images found for {category}/{defect_class}.")
                print(f"No images found for {category}/{defect_class}.")

        if class_kid_scores:
            category_mean_kid = np.mean(class_kid_scores)
            category_kid_scores[category] = category_mean_kid * 1000
            print(f"Mean KID score for {category}: {category_mean_kid}")
            logging.info(f"Mean KID score for {category}: {category_mean_kid}")
        else:
            logging.warning(f"No valid images for category {category}.")
            print(f"No valid images for category {category}.")

    print("\nMean KID score per category:")
    logging.info("Mean KID score per category:")
    for category, kid in category_kid_scores.items():
        print(f"{category}: {kid}")
        logging.info(f"{category}: {kid}")

    if category_kid_scores:
        overall_mean_kid = np.mean(list(category_kid_scores.values()))
        print(f"\nOverall mean KID score: {overall_mean_kid}")
        logging.info(f"Overall mean KID score: {overall_mean_kid}")

if __name__ == "__main__":
    main()