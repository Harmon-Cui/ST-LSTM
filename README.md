# Short-term Passenger Flow Prediction for Urban Rail Transit

This repository contains the source code and example data for the study titled:

**Short-term passenger flow prediction for urban rail systems: A deep learning approach utilizing multi-source big data.**

---

## Abstract

Predicting short-term passenger flow in urban rail transit is crucial for intelligent and real-time management of urban rail systems.  
This study utilizes deep learning techniques and multi-source big data to develop an enhanced spatial-temporal long short-term memory (ST-LSTM) model for forecasting subway passenger flow. The model includes three key components:

1. **Temporal correlation learning module**: captures travel patterns across stations to select effective training data.  
2. **Spatial correlation learning module**: extracts spatial correlations between stations using geographic information and passenger flow variations, providing interpretable quantification of these correlations.  
3. **Fusion module**: integrates historical spatial-temporal features with real-time data to accurately predict passenger flow.

The model is evaluated on two large-scale real-world subway datasets from Nanjing and Chongqing, showing superior performance over benchmarks.

---

## Repository Contents

| File name                                                  | Description                                                   |
|------------------------------------------------------------|---------------------------------------------------------------|
| `Article algorithm code.py`                                | Main Python code implementing the proposed ST-LSTM algorithm |
| `Distance coefficient between stations (Example).csv`      | Sample spatial distance data between stations (example data) |
| `Flow Coefficient Between Stations (Example).csv`          | Sample flow coefficient data (example data)                  |
| `Pearson Correlation Between Stations (Example).csv`       | Sample Pearson correlation data between stations (example data) |
| `Sample data of outbound passenger flow.csv`               | Sample passenger flow time series data (example data)        |
| `Stations to be predicted (Example).csv`                   | List of stations for which flow is predicted (example data)  |
| `Outbound volume prediction value.csv`                     | Output results of the passenger flow prediction              |
| `Prediction accuracy of outbound volume.csv`               | Prediction accuracy metrics (RMSE, MAE) output by the code   |
| `Screenshot of the code running result.png`                | Screenshot showing successful code execution                 |

---

## Important Notes on Data

Due to ethical and legal restrictions, **real subway AFC card data cannot be publicly shared**.  
The included example data files are **synthetic and only simulate the format and structure of the real data** used in the study.  
This ensures you can test and run the code smoothly, but results will not reflect real passenger flow patterns.

---

## Environment Setup

- Python 3.10+  
- Required Python packages can be installed with:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Place the main code file (`Article algorithm code.py`) and all example data files into the **same folder**.  
2. Ensure the environment is set up as described above.  
3. Run the main script using:

```bash
python "Article algorithm code.py"
```

4. After execution, the following output files will be generated:
   - `Outbound volume prediction value.csv`
   - `Prediction accuracy of outbound volume.csv`

5. You can refer to `Screenshot of the code running result.png` for a visual confirmation of a successful run.

---

## Citation

If you use this code or data in your research, please cite:

> Hongmeng Cui, Bingfeng Si, Dazhuang Chi, Yueqing Li, Ge Li, Yuanmeng Chen.  
> *"Short-term passenger flow prediction for urban rail systems: A deep learning approach utilizing multi-source big data,"*  
> (Under review, PLOS ONE, 2025)

---

## Contact

For questions, suggestions, or data access requests, please contact:

**Hongmeng Cui**  
Email: 13654081957@163.com  
Institution: Beijing Jiaotong University

---

## License

This project is licensed under the MIT License.  
See the `LICENSE` file for more details.
