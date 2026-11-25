# SmartPulse AI Lab: Congestion-Aware Emergency Ambulance Router

An AI Lab project combining **Supervised Machine Learning (K-Nearest Neighbors)** trained on real **Kaggle** traffic sensor data with **Informed Graph Search (A\* Search vs. BFS)** to route an emergency ambulance (`Depot -> Patient -> Hospital`) around predicted traffic congestion.

## Project Structure

```text
SmartPulse_AI_Lab/
├── data/
│   ├── Traffic.csv
│   └── TrafficTwoMonth.csv          # Downloaded automatically from kaggle.com
├── models/
│   └── knn_traffic_model.pkl        # Trained KNN classifier + StandardScaler
├── outputs/
│   ├── knn_evaluation.png           # K-tuning curve + Confusion Matrix
│   └── route_comparison.png         # City traffic maps + A* vs BFS chart
├── src/
│   ├── download_data.py             # Kaggle dataset downloader
│   ├── knn_model.py                 # Feature engineering & KNN training
│   ├── city_router.py               # 16x16 grid, KNN edge costing, A* & BFS
│   └── pdf_report.py                # Automated ReportLab PDF builder
├── tests/
│   └── test_project.py              # Unit & integration tests
├── main.py                          # Main CLI pipeline
├── requirements.txt
└── SmartPulse_AI_Project_Report.pdf # Full report & code explanation
```

## How to Run

1. Open PowerShell or Terminal and navigate to the project folder:
   ```powershell
   cd "$env:USERPROFILE\Desktop\SmartPulse_AI_Lab"
   ```
2. *(Optional)* Install dependencies if needed:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the full pipeline (downloads Kaggle dataset, trains KNN, executes A\* & BFS, creates plots and `SmartPulse_AI_Project_Report.pdf`):
   ```powershell
   python main.py
   ```
4. Try custom scenarios or flags:
   ```powershell
   python main.py --hour 18 --day Friday
   python main.py --force-download
   ```
5. Run automated tests:
   ```powershell
   python -m unittest discover -s tests -v
   ```
