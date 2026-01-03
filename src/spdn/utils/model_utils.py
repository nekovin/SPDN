import torch
from spdn.models.spdn.spdn_attention_small import get_spdn_model_simple
from spdn.models.spdn.spdn_attention import get_spdn_model_attention
from spdn.models.spdn.spdn_no_attention import get_spdn_model_no_attention


def load_ssm_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_spdn_model_attention(checkpoint=checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_spdn_model(
    model_name, train_config, loss_name, num_epochs, learning_rate, optim, checkpoint
):
    history = {"loss": [], "flow_loss": [], "noise_loss": []}

    if model_name == "SSMSimple":
        if train_config["load_model"]:
            checkpoint_path = train_config["best_checkpoint"].format(
                model_name=model_name, loss_fn=loss_name
            )
            model, checkpoint = get_spdn_model_simple(checkpoint_path=checkpoint_path)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_loss = checkpoint["best_loss"]
            set_epoch = checkpoint["epoch"]
            history = checkpoint["history"]
            num_epochs = num_epochs + set_epoch
            print(
                f"Model loaded from {checkpoint_path} at epoch {set_epoch} with loss {best_loss:.6f}"
            )
        else:
            model, checkpoint = get_spdn_model_simple(checkpoint_path=None)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            best_loss = float("inf")
            set_epoch = 0

    elif model_name == "SSMAttention":
        if train_config["load_model"]:
            checkpoint_path = train_config["best_checkpoint"].format(
                model_name=model_name, loss_fn=loss_name
            )
            model, checkpoint = get_spdn_model_attention(
                checkpoint_path=checkpoint_path
            )
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_loss = checkpoint["best_loss"]
            set_epoch = checkpoint["epoch"]
            history = checkpoint["history"]
            num_epochs = num_epochs + set_epoch
            print(
                f"Model loaded from {checkpoint_path} at epoch {set_epoch} with loss {best_loss:.6f}"
            )
        else:
            model, checkpoint = get_spdn_model_attention(checkpoint_path=None)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            best_loss = float("inf")
            set_epoch = 0

    elif model_name == "SPDNNoAttention":
        if train_config["load_model"]:
            checkpoint_path = train_config["best_checkpoint"].format(
                model_name=model_name, loss_fn=loss_name
            )
            model, checkpoint = get_spdn_model_no_attention(
                checkpoint_path=checkpoint_path
            )
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_loss = checkpoint["best_loss"]
            set_epoch = checkpoint["epoch"]
            history = checkpoint["history"]
            num_epochs = num_epochs + set_epoch
            print(
                f"Model loaded from {checkpoint_path} at epoch {set_epoch} with loss {best_loss:.6f}"
            )
        else:
            model, checkpoint = get_spdn_model_no_attention(checkpoint_path=None)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            best_loss = float("inf")
            set_epoch = 0

    else:
        raise ValueError(f"Model {model_name} is not recognised.")
    print(
        f"Model {model_name} initialized with learning rate {learning_rate} and optimizer {optim.__name__}"
    )

    return model, optimizer, best_loss, set_epoch, num_epochs, history
