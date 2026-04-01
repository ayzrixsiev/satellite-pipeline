import torch
import cv2
import numpy as np


def save_prediction(model, dataset, index, device, filename="result.png"):
    model.eval()

    image, mask = dataset[index]

    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()

    probs = torch.sigmoid(output)
    prediction = probs.squeeze().cpu().numpy()
    prediction_binary = (prediction > 0.5).astype(np.uint8) * 255

    original_image = image.cpu().numpy()
    original_image = np.transpose(original_image, (1, 2, 0))
    original_image = (original_image * 255).astype(np.uint8)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    prediction_bgr = cv2.cvtColor(prediction_binary, cv2.COLOR_GRAY2BGR)
    comparison_image = np.hstack([original_image, prediction_bgr])

    cv2.imwrite(filename, comparison_image)
    print(f"Prediction saved to {filename}")
