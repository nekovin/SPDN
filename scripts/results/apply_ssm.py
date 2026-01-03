import torch


def apply_model_to_dataset(
    model_path,
    dataset,
    batch_size=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Load a trained SpeckleSeparationModule and apply it to the input data

    Args:
        model_path: Path to the saved model
        input_target_data: List of tuples (input, target) where both are numpy arrays
        batch_size: Batch size for processing
        device: Device to run inference on ('cuda' or 'cpu')

    Returns:
        Dictionary containing denoised images, flow components, and noise components
    """
    print(f"Using device: {device}")

    # Load the model
    model = SpeckleSeparationModule(input_channels=1, feature_dim=32)
    # model = SpeckleSeparationUNet(input_channels=1, feature_dim=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare input tensors
    input_tensors = []

    dataset_list = []
    for i in dataset.keys():
        dataset_list.append(dataset[i])

    """
    for input_img, _ in input_target_data:  # Unpack the tuple, ignore the target
        # Convert to tensor and add channel dimension if needed
        if len(input_img.shape) == 2:
            input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)
        else:
            input_tensor = torch.from_numpy(input_img).float()
            
        input_tensors.append(input_tensor)"""

    flattened_dataset = []
    for patient_data in dataset_list:
        if isinstance(patient_data, list):
            flattened_dataset.extend(patient_data)
        else:
            flattened_dataset.append(patient_data)

    input_tensors = []
    for input_img in flattened_dataset:
        # Convert to tensor and add channel dimension if needed
        if isinstance(input_img, list):
            input_img = np.array(input_img)

        if len(input_img.shape) == 2:
            # For 2D images, add a single channel
            input_tensor = (
                torch.from_numpy(input_img).float().unsqueeze(0)
            )  # [H, W] -> [1, H, W]
        elif len(input_img.shape) == 3:
            # For 3D images, ensure we're using only the first channel if there are multiple
            if (
                input_img.shape[0] > 1 or input_img.shape[2] > 1
            ):  # Checking if first or last dim is channel
                # Take only the first channel
                if input_img.shape[0] > 1:  # If channels are in first dimension
                    input_img = input_img[0:1, :, :]
                elif input_img.shape[2] > 1:  # If channels are in last dimension
                    input_img = input_img[:, :, 0:1]
                    input_img = np.transpose(
                        input_img, (2, 0, 1)
                    )  # [H, W, 1] -> [1, H, W]
            input_tensor = torch.from_numpy(input_img).float()
        else:
            # Handle unexpected dimensions
            raise ValueError(f"Unexpected image shape: {input_img.shape}")

        input_tensors.append(input_tensor)

    # Stack into batch dimension
    inputs = torch.stack(input_tensors).to(device)

    # Create dataloader
    dataset = TensorDataset(inputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Results containers
    results = {
        "raw_image": [],
        "denoised_images": [],
        "flow_components": [],
        "noise_components": [],
    }

    # Process the dataset
    with torch.no_grad():
        for (batch_inputs,) in tqdm(dataloader, desc="Processing dataset"):
            # Apply model
            outputs = model(batch_inputs)

            raw_image = batch_inputs
            flow_component = outputs["flow_component"]
            flow_component = normalize_image(flow_component.cpu().numpy())
            noise_component = outputs["noise_component"]
            denoised = batch_inputs - noise_component

            # Store results (move to CPU and convert to numpy)
            results["raw_image"].append(raw_image.cpu().numpy())
            results["denoised_images"].append(denoised.cpu().numpy())
            # results['flow_components'].append(flow_component)
            # flow_component = threshold_octa(flow_component, method='percentile', threshold_percentage=0.01)
            results["flow_components"].append(flow_component)
            results["noise_components"].append(noise_component.cpu().numpy())

    # Concatenate batches
    for key in results:
        results[key] = np.concatenate(results[key], axis=0)

    print("Length of results:")
    for key in results:
        print(f"{key}: {len(results[key])}")

    return results
