from docx import Document
from docx.shared import Inches, Cm
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from datetime import datetime
import pandas as pd
import json
import os


def append_docx(master: Document, doc_to_append: Document):
    """
    Appende il contenuto di doc_to_append alla fine di master
    """
    for element in doc_to_append.element.body:
        master.element.body.append(element)


def replace_placeholders(doc, replacements: dict):
    """
    Sostituisce {{PLACEHOLDER}} nel documento (body + header + footer)
    """
    def replace_in_paragraphs(paragraphs):
        for p in paragraphs:
            for key, value in replacements.items():
                if key in p.text:
                    inline = p.runs
                    for run in inline:
                        for k, v in replacements.items():
                            if k in run.text:
                                run.text = run.text.replace(k, v)

    # Corpo documento
    replace_in_paragraphs(doc.paragraphs)

    # Header e footer
    for section in doc.sections:
        replace_in_paragraphs(section.header.paragraphs)
        replace_in_paragraphs(section.footer.paragraphs)


# --------------------------------------------------
# Load analysis outputs
# --------------------------------------------------
kpis = json.load(open("outputs/kpis.json"))
summary_df = pd.read_csv("outputs/summary_table.csv")
metrics_df = pd.read_csv("outputs/model_metrics.csv")

# --------------------------------------------------
# Create document
# --------------------------------------------------
replacements = {
    "{{data}}": datetime.today().strftime("%d/%m/%y"),
    "{{MESE_ANNO}}": "GENNAIO 2026",
    "{{OPERA}}": "P005 - PONTE ADIGE EST",
    "{{Comune}}": "VERONA (VR)"
}

doc = Document("templates/A4_P005_Adige_Ovest_Copertina.docx")
replace_placeholders(doc, replacements)

doc.add_page_break()

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
    cols=len(summary_df.columns)
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
    cols=len(metrics_df.columns)
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
output_path = os.path.join("outputs", "monthly_report.docx")
doc.save(output_path)
