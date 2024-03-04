# MED: Non-Autoregressive Generation-Based Real-Time Production Forecasting with Variable Input Length

Masked Encoding Decoding (MED) employs a non-autoregressive generation (NAG) method for production forecasting, offering a distinct approach to sequential data generation that allows for dynamic updates in time-series prediction.

## Citation

If you use this code in your research, please cite the following paper:

Xu, Z., & Leung, J. Y. (2024). Dynamic Real-Time Production Forecasting Model for Complex Subsurface Flow Systems with Variable-Length Input Sequences.

## Introduction

The introduction section is identical to that of the MRA method. The primary distinction with MED lies in its use of a non-autoregressive generation method, allowing for simultaneous prediction of the entire sequence without the iterative approach required by autoregressive methods.

## Methodology

### Model Structure

Similar to MRA, MED utilizes an encoder-decoder architecture. However, in contrast, the encoder in MED encodes historical production information into a vector, initializing the decoder’s hidden state for a single-pass generation of the entire forecast sequence, leveraging known operational data.

![Figure2](https://github.com/ziming-zx/MED/assets/55851734/ac6b450f-6c63-4655-a032-53a31e971cbf)


### Segmentation

While MED shares the flexible segmentation capability of MRA, accommodating time series of varying lengths, its approach to training pair formation is distinct. MED directly uses each data point (e.g., `x1`, `x2`, `x3`) to predict the entire sequence (`x1, x2, ..., xT`) in a single run. This results in `T` training pairs for each sequence, cumulating in `VxT` pairs across the dataset, where `V` is the total number of wells.

### Input and Output Shape

- **Encoder Input Shape:** Input is formatted as `[VxT, T, feature_number]`, accommodating `V` wells, `T` time steps per sequence, and a set number of features per time step, including static, dynamic, and autoregressive terms.

- **Decoder Input Shape:** Similarly structured as `[VxT, T, feature_number]`, it incorporates `V` wells and `T` time steps, focusing on static and dynamic features for each step.

- **Output Shape:** The output, formatted as `[VxT, T, 1]`, reflects the model’s capability to predict the entire sequence in one pass.

## Experiment Results

The performance of MED was evaluated across 200 wells in the testing dataset, with RMSE computed and analyzed for prediction horizons of `l = 0, 1, 3, and 5 months`. The results for P10, P50, and P90 cases demonstrate MED's superior performance compared to MRA, highlighting its efficiency in handling variable input lengths and its effectiveness in forecasting production time series.

![Figure6](https://github.com/ziming-zx/MED/assets/55851734/d97ebe92-20c6-4813-9a69-f90ef2205c17)

