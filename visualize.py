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

    prediction_binary = (prediction > 0.5).astype(np.uint8) * 255

    # 5. Save it
    cv2.imwrite(filename, prediction_binary)
    print(f"Prediction saved to {filename}")
