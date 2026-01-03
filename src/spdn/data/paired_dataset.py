import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from spdn.utils import paired_preprocessing


class PairedOCTDataset(Dataset):
    def __init__(
        self,
        start,
        n_patients=2,
        n_images_per_patient=50,
        transform=None,
        diabetes_list=[0, 1, 2],
    ):
        self.transform = transform
        dataset_dict = paired_preprocessing(
            start, n_patients, n_images_per_patient, diabetes_list=diabetes_list
        )

        self.input_images = []
        self.target_images = []

        for patient_id, data in dataset_dict.items():
            print(f"Processing patient {patient_id} with {len(data)} images")
            for i in range(len(data)):
                input_image = data[i][0]
                target_image = data[i][1]

                if input_image.shape != target_image.shape:
                    print(
                        f"Shape mismatch: {input_image.shape} vs {target_image.shape}"
                    )
                    continue

                if np.isfinite(input_image).all() and np.isfinite(target_image).all():
                    self.input_images.append(input_image)
                    self.target_images.append(target_image)

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_img = self.input_images[idx]
        target_img = self.target_images[idx]

        if len(input_img.shape) == 2:
            input_img = input_img[:, :, np.newaxis]
            target_img = target_img[:, :, np.newaxis]

        input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float()
        target_tensor = torch.from_numpy(target_img.transpose(2, 0, 1)).float()

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor


def get_paired_loaders(
    start,
    n_patients=2,
    n_images_per_patient=50,
    batch_size=8,
    val_split=0.2,
    shuffle=True,
    random_seed=42,
):
    full_dataset = PairedOCTDataset(
        start, n_patients=n_patients, n_images_per_patient=n_images_per_patient
    )

    dataset_size = len(full_dataset)
    print(f"Dataset size: {dataset_size}")
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    random = False
    if random:
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
    else:
        train_dataset = torch.utils.data.Subset(full_dataset, np.arange(train_size))

        val_dataset = torch.utils.data.Subset(
            full_dataset, np.arange(train_size, dataset_size)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
    )

    return train_loader, val_loader
