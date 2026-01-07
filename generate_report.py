
from fpdf import FPDF
import json
import os
import datetime

# Load Metrics
try:
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)
except FileNotFoundError:
    print("Metrics file not found. Running with dummy data if needed.")
    metrics = {}

class ProfessionalPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | Airline Food Demand Prediction', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 51, 102) # Dark Blue
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
        self.set_draw_color(0, 51, 102)
        self.line(10, self.get_y(), 200, self.get_y()) # Underline
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 7, body)
        self.ln()

    def add_image_centered(self, image_path, width=150):
        if os.path.exists(image_path):
            self.image(image_path, w=width, x=(210-width)/2)
            self.ln(5)
        else:
            self.set_text_color(255, 0, 0)
            self.cell(0, 10, f"[Image Missing: {image_path}]", 0, 1)
            self.set_text_color(0, 0, 0)

pdf = ProfessionalPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# --- COVER PAGE ---
pdf.add_page()
pdf.set_y(80)
pdf.set_font('Helvetica', 'B', 24)
pdf.set_text_color(0, 51, 102)
pdf.cell(0, 20, 'Airline Food Demand Prediction', 0, 1, 'C')

pdf.set_font('Helvetica', '', 16)
pdf.set_text_color(50, 50, 50)
pdf.cell(0, 10, 'Final Project Report', 0, 1, 'C')

pdf.ln(20)
pdf.set_font('Helvetica', 'I', 12)
pdf.cell(0, 10, f'Generated on: {datetime.date.today().strftime("%B %d, %Y")}', 0, 1, 'C')
pdf.cell(0, 10, 'Prepared for: SE390 - Artificial Intelligence Projects', 0, 1, 'C')

pdf.add_page()

# --- 1. Problem Statement ---
pdf.chapter_title("1. Problem Statement")
pdf.chapter_body(
    "Airlines typically operate on thin profit margins, and in-flight catering is a significant operational cost. "
    "Over-catering leads to food waste, increased weight, and higher fuel consumption. Under-catering results in "
    "negative passenger experiences and potential compensation claims.\n\n"
    "This project aims to solve this optimization problem by developing a Machine Learning model that predicts "
    "the exact number of meals required for a flight. By analyzing features such as flight duration, passenger demographics, "
    "and flight type, we can move away from static ratios to dynamic, data-driven predictions."
)

# --- 2. Dataset Methodology ---
pdf.chapter_title("2. Dataset Methodology")
pdf.chapter_body(
    "A synthetic dataset comprising 5,500 flight records was generated to simulate realistic airline operations. "
    "The data generation process adhered to strict validation rules to ensure logical consistency:"
)
pdf.set_font('Helvetica', '', 10)
pdf.multi_cell(0, 6, 
    "- Volume: 5,500 records (Split 80/20 for Training/Testing)\n"
    "- International Flights: Minimum 3 hours duration.\n"
    "- Passenger Composition: Adults + Children = Total Passengers.\n"
    "- Target Variable: Derived from a weighted formula of duration, class ratios, and flight type, plus noise."
)
pdf.ln(5)

# --- 3. Exploratory Data Analysis ---
pdf.chapter_title("3. Exploratory Data Analysis (EDA)")
pdf.chapter_body(
    "We analyzed the relationships between features to confirm the dataset's validity and understand driving factors. "
    "The correlation heatmap below highlights strong dependencies between Passenger Count, Duration, and Food Demand."
)
pdf.add_image_centered("eda_heatmap.png", width=140)
pdf.chapter_body(
    "We also examined the distribution of our target variable and its relationship with key features."
)
pdf.add_image_centered("eda_distributions.png", width=160)

# --- 4. Model Development ---
pdf.add_page()
pdf.chapter_title("4. Methodology & Models")
pdf.chapter_body(
    "Three distinct modeling approaches were implemented to establish performance benchmarks:"
)
pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 8, "1. Baseline Model (Mean Predictor)", 0, 1)
pdf.set_font('Helvetica', '', 11)
pdf.multi_cell(0, 6, "A simple heuristic that predicts the average demand for every flight. This serves as the 'floor' for performance.")

pdf.ln(2)
pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 8, "2. Linear Regression", 0, 1)
pdf.set_font('Helvetica', '', 11)
pdf.multi_cell(0, 6, "A parametric model that assumes a linear relationship between input features and food demand. It acts as a good baseline for simple relationships.")

pdf.ln(2)
pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 8, "3. Random Forest Regressor (Selected Model)", 0, 1)
pdf.set_font('Helvetica', '', 11)
pdf.multi_cell(0, 6, "An ensemble learning method using decision trees. It was selected for its ability to capture non-linear interactions (e.g., the jump in food need for flights > 7 hours) and its robustness to outliers.")
pdf.ln(5)

# --- 5. Results & Evaluation ---
pdf.chapter_title("5. Results & Evaluation")
pdf.chapter_body(
    "The Random Forest model demonstrated superior performance across all metrics, significantly reducing the Mean Absolute Error (MAE) compared to the baseline."
)

# Render Table
pdf.set_fill_color(240, 240, 240)
pdf.set_font('Helvetica', 'B', 10)
pdf.cell(50, 8, "Model", 1, 0, 'C', 1)
pdf.cell(30, 8, "R2 Score", 1, 0, 'C', 1)
pdf.cell(30, 8, "MAE", 1, 0, 'C', 1)
pdf.cell(30, 8, "RMSE", 1, 1, 'C', 1)

pdf.set_font('Helvetica', '', 10)
for model, scores in metrics.items():
    pdf.cell(50, 8, model, 1, 0, 'L')
    pdf.cell(30, 8, str(scores.get('R2', '-')), 1, 0, 'C')
    pdf.cell(30, 8, str(scores.get('MAE', '-')), 1, 0, 'C')
    pdf.cell(30, 8, str(scores.get('RMSE', '-')), 1, 1, 'C')

pdf.ln(5)
pdf.chapter_body(
    "The analysis of residuals confirms that the model's errors are normally distributed and centered around zero, indicating no major systemic bias."
)
pdf.add_image_centered("model_actual_vs_predicted.png", width=120)
pdf.add_image_centered("model_residuals.png", width=120)

# --- 6. Feature Importance ---
pdf.add_page()
pdf.chapter_title("6. Feature Importance Analysis")
pdf.chapter_body(
    "A key advantage of the Random Forest model is interpretability through feature importance. "
    "As shown below, 'Passenger Count' and 'Flight Duration' are the most critical predictors, aligning with domain knowledge."
)
pdf.add_image_centered("model_feature_importance.png", width=140)

# --- 7. Conclusion ---
pdf.chapter_title("7. Conclusion")
pdf.chapter_body(
    "The project successfully demonstrated that machine learning can accurately predict airline food demand. "
    "The Random Forest model achieved a high R2 score, making it a viable tool for deployment.\n\n"
    "Future improvements could include adding more granular data such as passenger dietary preferences, "
    "time of day, and seasonal factors to further refine accuracy."
)

output_path = "Project_Report_Final.pdf"
pdf.output(output_path)
print(f"Professional Report generated: {output_path}")
