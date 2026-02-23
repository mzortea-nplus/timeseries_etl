from docx import Document
from docx.shared import Cm
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd
import json
import os
from docx.shared import Cm
from docx.shared import Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docxcompose.composer import Composer


# --------------------------------------------------
# Utility
# --------------------------------------------------


def set_font(doc, font_name="Times New Roman", font_size=12):
    # Paragrafi
    for p in doc.paragraphs:
        for run in p.runs:
            run.font.name = font_name
            run.font.size = Pt(font_size)
            run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)

    # Tabelle
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.name = font_name
                        run.font.size = Pt(font_size)
                        run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)

    # Stili
    for style_name in ["Normal", "Title", "Heading 1", "Heading 2"]:
        style = doc.styles[style_name]
        style.font.name = font_name
        style.font.size = Pt(font_size)


def replace_placeholders(doc, replacements: dict):

    def replace_in_paragraphs(paragraphs):
        for p in paragraphs:
            for run in p.runs:
                for k, v in replacements.items():
                    if k in run.text:
                        run.text = run.text.replace(k, v)

    # Corpo documento
    replace_in_paragraphs(doc.paragraphs)

    # Tabelle nel corpo
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                replace_in_paragraphs(cell.paragraphs)

    # Header e footer
    for section in doc.sections:
        replace_in_paragraphs(section.header.paragraphs)
        replace_in_paragraphs(section.footer.paragraphs)

        for table in section.header.tables:
            for row in table.rows:
                for cell in row.cells:
                    replace_in_paragraphs(cell.paragraphs)

        for table in section.footer.tables:
            for row in table.rows:
                for cell in row.cells:
                    replace_in_paragraphs(cell.paragraphs)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

cliente = "A4"
template_path = "templates/A4_Template.docx"

opere = {
    "P001_Sommacampagna": {
        "label": "P001 - SOMMACAMPAGNA",
        "comune": "SOMMACAMPAGNA (VR)",
    },
    "P002_Giuliari_Milani": {
        "label": "P002 - GIULIARI MILANI",
        "comune": "VERONA (VR)",
    },
    "P003_Gua": {"label": "P003 - GUA", "comune": "GUA (VR)"},
    "P004_Adige_Est": {"label": "P004 - ADIGE EST", "comune": "VERONA (VR)"},
    "P005_Adige_Ovest": {"label": "P005 - ADIGE OVEST", "comune": "VERONA (VR)"},
}


start_period = date(2025, 10, 1)
end_period = date(2026, 2, 1)  # esclusivo

MESI_IT = {
    1: "GENNAIO",
    2: "FEBBRAIO",
    3: "MARZO",
    4: "APRILE",
    5: "MAGGIO",
    6: "GIUGNO",
    7: "LUGLIO",
    8: "AGOSTO",
    9: "SETTEMBRE",
    10: "OTTOBRE",
    11: "NOVEMBRE",
    12: "DICEMBRE",
}


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

for opera_key, opera_info in opere.items():
    opera_label = opera_info["label"]
    opera_comune = opera_info["comune"]

    current = start_period

    while current < end_period:

        year = current.year
        month = current.month

        mese_tag = f"{year}_{month:02d}"
        mese_nome = f"{MESI_IT[month]} {year}"

        start_month = current
        end_month = current + relativedelta(months=1)

        # print(f"Generazione report {opera_key} - {mese_tag}")

        # --------------------------------------------------
        # Paths
        # --------------------------------------------------

        base_out = os.path.join("outputs", opera_key, mese_tag)
        fig_dir = os.path.join("figures", opera_key, mese_tag)

        if not os.path.isdir(base_out):
            print(f"  ⚠ dati mancanti, salto {mese_tag}")
            current += relativedelta(months=1)
            continue

        # --------------------------------------------------
        # Load data
        # --------------------------------------------------

        with open(os.path.join(base_out, "kpis.json"), encoding="utf-8") as f:
            kpis = json.load(f)

        summary_df = pd.read_csv(os.path.join(base_out, "summary_table.csv"))
        metrics_df = pd.read_csv(os.path.join(base_out, "model_metrics.csv"))

        ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # if filename.endswith("_x") or filename.endswith('_y'):
        #     sens_type = 'inclinometer'
        # elif filename.endswith("t"):
        ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        figures = []
        if os.path.isdir(fig_dir):
            figures = sorted(
                [
                    f
                    for f in os.listdir(fig_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )

        # --------------------------------------------------
        # Create document
        # --------------------------------------------------

        replacements = {
            "Cliente": cliente,
            "{{data}}": datetime.today().strftime("%d/%m/%Y"),
            "{{MESE_ANNO}}": mese_nome,
            "{{PERIODO_DAL}}": start_month.strftime("%d/%m/%Y"),
            "{{PERIODO_AL}}": end_month.strftime("%d/%m/%Y"),
            "{{OPERA}}": opera_label,
            "{{Comune}}": opera_comune,
        }

        master = Document(template_path)
        # doc = Document()
        # set_font(doc, "Times New Roman", 12)
        # doc.styles['Normal'].font.name = 'Times New Roman'

        replace_placeholders(master, replacements)

        # --------------------------------------------------
        # Title page
        # --------------------------------------------------

        # title = master.add_heading("Monthly Structural Monitoring Report", 1)
        # title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        # master.add_paragraph(f"Reporting period: {mese_nome}")
        # master.add_paragraph("Generated automatically")
        # master.add_page_break()


        # --------------------------------------------------
        # Section 1: Executive Summary
        # --------------------------------------------------

        # master.add_heading("1. Executive Summary", level=1)
        # p = master.add_paragraph()
        # p.add_run(f"Average displacement: {kpis['avg_displacement']} mm\n")
        # p.add_run(f"Max acceleration: {kpis['max_acceleration']} m/s²\n")
        # p.add_run(f"Detected anomalies: {kpis['n_anomalies']}")

        composer = Composer(master)
        description = Document(fr"templates/description/{opera_label[:4]}_description.docx")
        composer.append(description)
        composer.save("temp.docx")  # Salva un documento temporaneo per evitare errori di composizione
        master.add_page_break()

        # --------------------------------------------------
        # Section 2: Summary Table
        # --------------------------------------------------

        master.add_heading("2. Summary Statistics", level=1)

        # larghezze
        col_widths = [Cm(3), Cm(2.5)]

        table = master.add_table(rows=1, cols=len(col_widths))
        table.autofit = False

        # forzo la larghezza delle colonne
        for i, width in enumerate(col_widths):
            table.columns[i].width = width
            # aggiunta robusta per Word
            for cell in table.columns[i].cells:
                tc = cell._tc
                tcPr = tc.get_or_add_tcPr()
                w = OxmlElement("w:tcW")
                w.set(qn("w:w"), str(int(width.pt * 30)))  # conversione punti -> twips
                w.set(qn("w:type"), "dxa")
                tcPr.append(w)

        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        for i, col in enumerate(summary_df.columns):
            table.rows[0].cells[i].text = col

        for _, row in summary_df.iterrows():
            row_cells = table.add_row().cells
            for i, val in enumerate(row):
                row_cells[i].text = str(val)

        master.add_page_break()

        # --------------------------------------------------
        # Section 3: Visual Analysis
        # --------------------------------------------------

        master.add_heading("3. Visual Analysis", level=1)

        # Filtra solo i grafici che iniziano con "sensor"
        sensor_figures = sorted([f for f in figures if f.startswith("raw")])

        if sensor_figures:

            master.add_paragraph(
                "Si riportano di seguito i grafici dei dati grezzi di tutti i sensori, "
                "relativi al periodo di riferimento di questo report. "
                "Si consideri l'associazione tra l'ID del sensore nel titolo del grafico "
                "e l'etichetta della posizione indicata negli elaborati grafici."
            )

            table = master.add_table(rows=1, cols=2)

            for i in range(0, len(sensor_figures), 2):
                row_cells = table.add_row().cells

                row_cells[0].paragraphs[0].add_run().add_picture(
                    os.path.join(fig_dir, sensor_figures[i]), width=Cm(9)
                )

                if i < len(sensor_figures) - 1:
                    row_cells[1].paragraphs[0].add_run().add_picture(
                        os.path.join(fig_dir, sensor_figures[i + 1]), width=Cm(9)
                    )

        else:
            master.add_paragraph("Non sono disponibili grafici dei sensori per il mese selezionato.")

        master.add_page_break()


        # --------------------------------------------------
        # Section 4: Model Performance
        # --------------------------------------------------

        master.add_heading("4. Model Performance", level=1)

        table = master.add_table(rows=1, cols=len(metrics_df.columns))

        for i, col in enumerate(metrics_df.columns):
            table.rows[0].cells[i].text = col

        for _, row in metrics_df.iterrows():
            row_cells = table.add_row().cells
            for i, val in enumerate(row):
                row_cells[i].text = f"{val:.3f}"

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        output_name = f"{cliente}_{opera_key}_{mese_tag}.docx"
        output_dir = os.path.join("outputs", opera_key, mese_tag)

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, output_name)
        master.save(output_path)

        master.save(output_path)
        print(f"\033[92m✔ salvato {output_name}\033[0m")

        current += relativedelta(months=1)
