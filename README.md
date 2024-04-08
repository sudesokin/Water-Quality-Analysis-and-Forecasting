# Water Quality Analysis and Prediction

This project focuses on analyzing and predicting water quality levels using various parameters that define its potability. Through an in-depth dataset and employing machine learning models, we aim to achieve accurate predictions to ensure water safety and quality.

## Table of Contents
- [About the Dataset](#about-the-dataset)
- [Scope of the Project](#scope-of-the-project)
- [Methods and Results](#methods-and-results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About the Dataset

The dataset comprises several parameters vital for assessing water's quality and safety for consumption:

- **pH Value**: Indicator of water's acid-base balance, within WHO recommended range of 6.5 to 8.5.
- **Hardness**: Measurement of calcium and magnesium salts, which determine water's ability to precipitate soap.
- **Solids (TDS)**: Reflects water's mineralization with a desirable limit set at 500 mg/L for drinking water.
- **Chloramines**: Disinfectants used in water, safe up to 4 mg/L.
- **Sulfate**: Naturally occurring substances with concentrations in freshwater typically ranging from 3 to 30 mg/L.
- **Conductivity**: Indicator of water's ionic concentration with a standard limit not exceeding 400 μS/cm as per WHO.
- **Organic Carbon**: Measures carbon in organic compounds in water, with EPA standards set for treated and source water.
- **Trihalomethanes (THMs)**: By-products of chlorine treatment, safe up to 80 ppm.
- **Turbidity**: Measure of water's clarity, with WHO standards recommending values below 5.00 NTU.
- **Potability**: Binary indicator of whether water is safe (1) or not (0) for consumption.

## Scope of the Project

The project entails a comprehensive approach to predict future water quality levels accurately, covering:
- Data collection
- Preprocessing
- Feature engineering
- Modeling phases

## Methods and Results

We utilized Random Forest, XGBoost, and LightGBM models in a streamlined pipeline, achieving:
- An 83% accuracy rate in forecasting water quality parameters.

## Conclusion

Our project demonstrates the effectiveness of machine learning techniques in predicting water quality, contributing towards ensuring the safety and potability of water.

## Contributing

Contributions are welcome to improve the accuracy of predictions and expand the dataset. Please refer to the contributing guidelines for more details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any inquiries or contributions, please contact us at [sudesokin@gmail.com](mailto:sudesokin@gmail.com).
