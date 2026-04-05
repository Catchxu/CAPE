import torch


def load_scbert_pretrained(model, checkpoint_path: str, logger):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    logger.info(
        "Loading %d compatible pretrained parameters from %s",
        len(compatible),
        checkpoint_path,
    )
    model_state.update(compatible)
    model.load_state_dict(model_state)
    return model
