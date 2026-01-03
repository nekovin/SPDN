from spdn.utils.data_utils.standard_preprocessing import standard_preprocessing
from spdn.utils.data_utils.oct_preprocessing import (
    octa_preprocessing,
    remove_speckle_noise,
)
from spdn.utils.data_utils.data_loading import load_patient_data
from spdn.utils.data_utils.helper import extract_number
import os
import random


def pair_data(preprocessed_data, octa_data, n_images_per_patient):
    n_neighbours = (len(preprocessed_data) - len(octa_data)) // 2

    input_target = []
    for i in range(len(octa_data)):
        oct_image = preprocessed_data[i + n_neighbours]
        octa_image = octa_data[i]
        input_target.append([oct_image, octa_image])

        if len(input_target) >= n_images_per_patient:
            break

    return input_target


def paired_octa_preprocessing(
    start=1,
    n_patients=1,
    n_images_per_patient=10,
    n_neighbours=2,
    threshold=0.65,
    sample=False,
    post_process_size=10,
):
    dataset = {}

    diabetes = 0
    base_data_path = os.environ["DATASET_DIR_PATH"]

    try:
        for i in range(start, start + n_patients):
            if diabetes != 0:
                base_data_path = +rf"{diabetes}\RawDataQA-{diabetes} ({i})"
            else:
                data_path = base_data_path + rf"0\RawDataQA ({i})"
            print(f"Processing patient {i}")
            data = load_patient_data(data_path)
            assert len(data) > 0, f"No data found for patient {i}"

            preprocessed_data = standard_preprocessing(data)

            octa_data = octa_preprocessing(preprocessed_data, n_neighbours, threshold)

            cleaned_octa_data = []
            for octa_img in octa_data:
                cleaned_img = remove_speckle_noise(octa_img, min_size=post_process_size)
                cleaned_octa_data.append(cleaned_img)

            input_target_data = pair_data(
                preprocessed_data, cleaned_octa_data, n_images_per_patient
            )

            dataset[i] = input_target_data

        return dataset

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


def _paired_preprocessing(start=1, n_patients=1, diabetes_list=[0, 1, 2], sample=False):
    """
    Preprocess OCT images without OCTA calculation, maintaining paired structure.
    """
    dataset = {}
    base_data_path = os.environ["DATASET_DIR_PATH"]

    try:
        for diabetes in diabetes_list:
            for i in range(start, start + n_patients):
                if diabetes != 0:
                    data_path = (
                        base_data_path + rf"{diabetes}\RawDataQA-{diabetes} ({i})"
                    )
                else:
                    data_path = base_data_path + rf"0\RawDataQA ({i})"
                print(f"Processing patient {i}")
                data = load_patient_data(data_path)
                print(f"Loaded {len(data)} images for patient {i}")
                assert len(data) > 0, f"No data found for patient {i}"

                preprocessed_data = standard_preprocessing(data)

                input_target = []
                for j in range(len(preprocessed_data) - 1):
                    image1 = preprocessed_data[j]
                    image2 = preprocessed_data[j + 1]
                    input_target.append([image1, image2])

                dataset[i] = input_target

        return dataset

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


def paired_preprocessing(start=1, n_patients=1, diabetes_list=[0, 1, 2], sample=False):
    dataset = {}
    base_data_path = os.environ["DATASET_DIR_PATH"]

    patient_count = 0  # Counter for total patients processed
    dataset_index = 0  # New counter for dataset keys

    try:
        for diabetes in diabetes_list:
            diabetes_path = os.path.join(base_data_path, f"{diabetes}")
            patient_dirs = sorted(os.listdir(diabetes_path), key=extract_number)
            sorted_patients = [
                os.path.join(diabetes_path, patient) for patient in patient_dirs
            ]
            random.shuffle(sorted_patients)

            for i, patient_path in enumerate(sorted_patients, 2):
                if patient_count >= n_patients:
                    break

                patient_id = extract_number(os.path.basename(patient_path))

                data = load_patient_data(patient_path)
                print(f"Loaded {len(data)} images for patient {patient_id}")
                assert len(data) > 0, f"No data found for patient {patient_id}"

                preprocessed_data = standard_preprocessing(data)

                if len(preprocessed_data) <= 1:
                    print(
                        f"Warning: Patient {patient_id} has insufficient images ({len(preprocessed_data)})"
                    )
                    continue

                print(f"Preprocessed data shape: {preprocessed_data.shape}")

                input_target = []
                for j in range(len(preprocessed_data) - 1):
                    image1 = preprocessed_data[j]
                    image2 = preprocessed_data[j + 1]

                    # Verify image shapes
                    if image1.shape != (256, 256, 1) or image2.shape != (256, 256, 1):
                        print(
                            f"WARNING: Unexpected image shape: {image1.shape}, {image2.shape}"
                        )

                    input_target.append([image1, image2])

                dataset_index += 1  # Use a sequential index for the dataset
                dataset[dataset_index] = input_target
                patient_count += 1

            if patient_count >= n_patients:
                break

        return dataset

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback

        traceback.print_exc()
        return None


def paired_preprocessing(
    start=1,
    n_patients=1,
    n_images_per_patient=10,
    diabetes_list=[0, 1, 2],
    sample=False,
):
    dataset = {}
    base_data_path = os.environ["DATASET_DIR_PATH"]

    dataset_index = 0

    try:
        all_patients = []
        for diabetes in diabetes_list:
            diabetes_path = os.path.join(base_data_path, f"{diabetes}")
            patient_dirs = sorted(os.listdir(diabetes_path), key=extract_number)

            for patient_dir in patient_dirs:
                patient_path = os.path.join(diabetes_path, patient_dir)
                all_patients.append((patient_path, diabetes))

        random.shuffle(all_patients)

        patients_per_category = n_patients // len(diabetes_list)
        remainder = n_patients % len(diabetes_list)

        selected_count = {diabetes: 0 for diabetes in diabetes_list}

        for patient_path, diabetes_type in all_patients:
            if selected_count[diabetes_type] >= patients_per_category + (
                1 if diabetes_type < remainder else 0
            ):
                continue

            patient_id = extract_number(os.path.basename(patient_path))

            data = load_patient_data(patient_path)
            data = data[:n_images_per_patient]
            print(
                f"Loaded {len(data)} images for patient {patient_id} (diabetes type {diabetes_type})"
            )

            if len(data) == 0:
                print(f"Warning: No data found for patient {patient_id}")
                continue

            preprocessed_data = standard_preprocessing(data)

            if len(preprocessed_data) <= 1:
                print(
                    f"Warning: Patient {patient_id} has insufficient images ({len(preprocessed_data)})"
                )
                continue

            print(f"Preprocessed data shape: {preprocessed_data.shape}")

            input_target = []
            available_indices = list(range(len(preprocessed_data) - 1))
            random.shuffle(available_indices)
            while len(input_target) < n_images_per_patient and available_indices:
                j = available_indices.pop(0)  # Take the next random index and remove it

                if j + 1 < len(preprocessed_data):  # Ensure we have a valid pair
                    image1 = preprocessed_data[j]
                    image2 = preprocessed_data[j + 1]

                    # Verify image shapes
                    if image1.shape != (256, 256, 1) or image2.shape != (256, 256, 1):
                        print(
                            f"WARNING: Unexpected image shape: {image1.shape}, {image2.shape}"
                        )
                        continue

                    input_target.append([image1, image2])

            dataset_index += 1  # Use a sequential index for the dataset
            dataset[dataset_index] = input_target

            # Update selected count for this diabetes type
            selected_count[diabetes_type] += 1

            # Check if we've selected enough patients total
            if sum(selected_count.values()) >= n_patients:
                break

        print(f"Selected patients by diabetes type: {selected_count}")
        return dataset

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback

        traceback.print_exc()
        return None


def paired_octa_preprocessing(
    start=1,
    n_patients=1,
    n_images_per_patient=10,
    n_neighbours=2,
    threshold=0.65,
    sample=False,
    post_process_size=10,
    diabetes_list=[0, 1, 2],
):
    dataset = {}
    base_data_path = os.environ["DATASET_DIR_PATH"]
    dataset_index = 0

    try:
        # Collect all available patients across diabetes categories
        all_patients = []
        for diabetes in diabetes_list:
            diabetes_path = os.path.join(base_data_path, f"{diabetes}")
            patient_dirs = sorted(os.listdir(diabetes_path), key=extract_number)

            for patient_dir in patient_dirs:
                patient_path = os.path.join(diabetes_path, patient_dir)
                all_patients.append((patient_path, diabetes))

        random.shuffle(all_patients)

        # Calculate distribution among diabetes categories
        patients_per_category = n_patients // len(diabetes_list)
        remainder = n_patients % len(diabetes_list)
        selected_count = {diabetes: 0 for diabetes in diabetes_list}

        for patient_path, diabetes_type in all_patients:
            if selected_count[diabetes_type] >= patients_per_category + (
                1 if diabetes_type < remainder else 0
            ):
                continue

            patient_id = extract_number(os.path.basename(patient_path))

            # Load and preprocess data
            data = load_patient_data(patient_path)
            print(
                f"Loaded {len(data)} images for patient {patient_id} (diabetes type {diabetes_type})"
            )
            if len(data) < n_neighbours + 1:
                print(
                    f"Warning: Patient {patient_id} has insufficient images ({len(data)})"
                )
                continue

            preprocessed_data = standard_preprocessing(data)
            if len(preprocessed_data) < n_neighbours + 1:
                print(
                    f"Warning: Patient {patient_id} has insufficient preprocessed images ({len(preprocessed_data)})"
                )
                continue

            # Create OCTA data
            octa_data = octa_preprocessing(preprocessed_data, 2, threshold)

            # Clean OCTA data
            cleaned_octa_data = []
            for octa_img in octa_data:
                cleaned_img = remove_speckle_noise(octa_img, min_size=post_process_size)
                cleaned_octa_data.append(cleaned_img)

            # Ensure we have cleaned OCTA data
            if len(cleaned_octa_data) == 0:
                print(
                    f"Warning: No cleaned OCTA data generated for patient {patient_id}"
                )
                continue

            # Create proper input-target pairs
            # The OCTA images should align with corresponding B-scans with n_neighbours offset
            input_target = []
            for i in range(len(cleaned_octa_data)):
                if i + n_neighbours < len(preprocessed_data):
                    oct_image = preprocessed_data[i + n_neighbours]
                    octa_image = cleaned_octa_data[i]

                    # Verify shapes
                    if oct_image.shape != (256, 256, 1):
                        print(f"WARNING: Unexpected OCT image shape: {oct_image.shape}")
                        continue

                    if octa_image.shape != (256, 256, 1):
                        print(
                            f"WARNING: Unexpected OCTA image shape: {octa_image.shape}"
                        )
                        continue

                    input_target.append([oct_image, octa_image])

                    if len(input_target) >= n_images_per_patient:
                        break

            if len(input_target) > 0:
                dataset_index += 1
                dataset[dataset_index] = input_target
                selected_count[diabetes_type] += 1

            if sum(selected_count.values()) >= n_patients:
                break

        print(f"Selected patients by diabetes type: {selected_count}")
        return dataset

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback

        traceback.print_exc()
        return None


def paired_octa_preprocessing_binary(
    start=1,
    n_patients=1,
    n_images_per_patient=10,
    n_neighbours=2,
    threshold=0.65,
    sample=False,
    post_process_size=10,
    diabetes_list=[0, 1, 2],
):
    dataset = {}
    base_data_path = os.environ["DATASET_DIR_PATH"]
    dataset_index = 0

    try:
        # Collect all available patients across diabetes categories
        all_patients = []
        for diabetes in diabetes_list:
            diabetes_path = os.path.join(base_data_path, f"{diabetes}")
            patient_dirs = sorted(os.listdir(diabetes_path), key=extract_number)

            for patient_dir in patient_dirs:
                patient_path = os.path.join(diabetes_path, patient_dir)
                all_patients.append((patient_path, diabetes))

        random.shuffle(all_patients)

        # Calculate distribution among diabetes categories
        patients_per_category = n_patients // len(diabetes_list)
        remainder = n_patients % len(diabetes_list)
        selected_count = {diabetes: 0 for diabetes in diabetes_list}

        for patient_path, diabetes_type in all_patients:
            if selected_count[diabetes_type] >= patients_per_category + (
                1 if diabetes_type < remainder else 0
            ):
                continue

            patient_id = extract_number(os.path.basename(patient_path))

            data = load_patient_data(patient_path)
            print(
                f"Loaded {len(data)} images for patient {patient_id} (diabetes type {diabetes_type})"
            )
            if len(data) < n_neighbours + 1:
                print(
                    f"Warning: Patient {patient_id} has insufficient images ({len(data)})"
                )
                continue

            preprocessed_data = standard_preprocessing(data)
            if len(preprocessed_data) < n_neighbours + 1:
                print(
                    f"Warning: Patient {patient_id} has insufficient preprocessed images ({len(preprocessed_data)})"
                )
                continue

            # Create OCTA data
            octa_data = octa_preprocessing(preprocessed_data, 2, threshold)

            # binary thresholding turn pixels to 0 or 1
            octa_data = [(octa_img > 0).astype("uint8") for octa_img in octa_data]

            # Clean OCTA data
            cleaned_octa_data = []
            for octa_img in octa_data:
                cleaned_img = remove_speckle_noise(octa_img, min_size=post_process_size)
                cleaned_octa_data.append(cleaned_img)

            # Ensure we have cleaned OCTA data
            if len(cleaned_octa_data) == 0:
                print(
                    f"Warning: No cleaned OCTA data generated for patient {patient_id}"
                )
                continue

            input_target = []
            for i in range(len(cleaned_octa_data)):
                if i + n_neighbours < len(preprocessed_data):
                    oct_image = preprocessed_data[i + n_neighbours]
                    octa_image = cleaned_octa_data[i]

                    # Verify shapes
                    if oct_image.shape != (256, 256, 1):
                        print(f"WARNING: Unexpected OCT image shape: {oct_image.shape}")
                        continue

                    if octa_image.shape != (256, 256, 1):
                        print(
                            f"WARNING: Unexpected OCTA image shape: {octa_image.shape}"
                        )
                        continue

                    input_target.append([oct_image, octa_image])

                    if len(input_target) >= n_images_per_patient:
                        break

            if len(input_target) > 0:
                dataset_index += 1
                dataset[dataset_index] = input_target
                selected_count[diabetes_type] += 1

            if sum(selected_count.values()) >= n_patients:
                break

        print(f"Selected patients by diabetes type: {selected_count}")
        return dataset

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback

        traceback.print_exc()
        return None
