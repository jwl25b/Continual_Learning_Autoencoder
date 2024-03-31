import torch
import numpy as np

def find_relatedness(reconstruction_error_autoencoder, reconstruction_error_data):
    return np.abs((reconstruction_error_autoencoder - reconstruction_error_data)/reconstruction_error_autoencoder)

def find_best_autoencoders(batchwise_data, autoencoders):
    with torch.no_grad():
        minimum_relatedness = float("inf")
        best_index=-1
        for index, autoencoder in autoencoders.items():
            prediction = autoencoder.get_prediction(batchwise_data)
            reconstruction_error_data = autoencoder.get_reduced_loss(prediction, batchwise_data)
            reconstruction_error_autoencoder = autoencoder.mean
            relatedness = find_relatedness(reconstruction_error_autoencoder, reconstruction_error_data)
            if relatedness.item()<minimum_relatedness:
                minimum_relatedness=relatedness
                best_index = index
        return best_index

def find_num_of_outliers(batchwise_data, model):
    with torch.no_grad():
        model_outputs = model.get_prediction(batchwise_data)
        unreduced_loss = model.get_unreduced_loss(model_outputs, batchwise_data)
        element_mean = torch.mean(unreduced_loss, axis=1)
        outliers = batchwise_data.shape[0] - sum([item>model.mean - 3 * model.std and item<model.mean + 3 * model.std for item in element_mean]).item()
        return outliers
