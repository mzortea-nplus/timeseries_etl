from docx import Document
from docx.shared import Inches, Cm
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
import pandas as pd
import json
import os
# --------------------------------------------------
# Load analysis outputs
# --------------------------------------------------
kpis = json.load(open("outputs/kpis.json"))
summary_df = pd.read_csv("outputs/summary_table.csv")
metrics_df = pd.read_csv("outputs/model_metrics.csv")

# --------------------------------------------------
# Create document
# --------------------------------------------------
doc = Document()

# Title page
title = doc.add_heading("Monthly Structural Monitoring Report", 0)
title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

doc.add_paragraph("Reporting period: January 2026")
doc.add_paragraph("Generated automatically")

doc.add_page_break()

# --------------------------------------------------
# Section 1: Executive Summary
# --------------------------------------------------
doc.add_heading("1. Executive Summary", level=1)

p = doc.add_paragraph()
p.add_run(f"Average displacement: {kpis['avg_displacement']} mm\n")
p.add_run(f"Max acceleration: {kpis['max_acceleration']} m/s²\n")
p.add_run(f"Detected anomalies: {kpis['n_anomalies']}")

doc.add_page_break()

# --------------------------------------------------
# Section 2: Summary Table
# --------------------------------------------------
doc.add_heading("2. Summary Statistics", level=1)

table = doc.add_table(
    rows=1,
    cols=len(summary_df.columns),
    style="Table Grid"
)
table.alignment = WD_TABLE_ALIGNMENT.CENTER

hdr_cells = table.rows[0].cells
for i, col in enumerate(summary_df.columns):
    hdr_cells[i].text = col

for _, row in summary_df.iterrows():
    row_cells = table.add_row().cells
    for i, val in enumerate(row):
        row_cells[i].text = str(val)

doc.add_page_break()

# --------------------------------------------------
# Section 3: Visual Analysis
# --------------------------------------------------
doc.add_heading("3. Visual Analysis", level=1)

paragraph = doc.add_paragraph("Long-term trend analysis:")
table = doc.add_table(
    rows=1,
    cols=2
)

figures = os.listdir('figures')
for i in range(0, len(figures), 2):
    row_cells = table.add_row().cells
    row_cells[0].paragraphs[0].add_run().add_picture(os.path.join('figures', figures[i]), width=Cm(7))
    if i < len(figures) - 1: 
        row_cells[1].paragraphs[0].add_run().add_picture(os.path.join('figures', figures[i+1]), width=Cm(7))

doc.add_page_break()

# --------------------------------------------------
# Section 4: Model Performance
# --------------------------------------------------
doc.add_heading("4. Model Performance", level=1)

table = doc.add_table(
    rows=1,
    cols=len(metrics_df.columns),
    style="Light Shading"
)

hdr_cells = table.rows[0].cells
for i, col in enumerate(metrics_df.columns):
    hdr_cells[i].text = col

for _, row in metrics_df.iterrows():
    row_cells = table.add_row().cells
    for i, val in enumerate(row):
        row_cells[i].text = f"{val:.3f}"

# --------------------------------------------------
# Save
# --------------------------------------------------
doc.save("monthly_report.docx")