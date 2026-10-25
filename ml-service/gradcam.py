import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

def make_gradcam(model, input_tensor, original_image: Image.Image, target_layer, class_index: int) -> Image.Image:
    activations, gradients = [], []
    def forward_hook(_module, _inputs, output):
        activations.append(output)
    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        logits[:, class_index].sum().backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(original_image.height, original_image.width), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0: cam /= cam.max()
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(np.asarray(original_image.convert("RGB")), 0.58, heat, 0.42, 0)
        return Image.fromarray(overlay)
    finally:
        h1.remove()
        h2.remove()
