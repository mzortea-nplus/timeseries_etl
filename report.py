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
from docx.shared import Inches


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

cliente = "A4"
template_path = "templates/A4_Template.docx"

opere = {
    "P001_Sommacampagna":   {"label": "P001 - SOMMACAMPAGNA","comune": "SOMMACAMPAGNA (VR)",},
    "P002_Giuliari_Milani": {"label": "P002 - GIULIARI MILANI","comune": "VERONA (VR)",},
    "P003_Gua":             {"label": "P003 - GUA", "comune": "GUA (VR)"},
    "P004_Adige_Est":       {"label": "P004 - ADIGE EST", "comune": "VERONA (VR)"},
    "P005_Adige_Ovest":     {"label": "P005 - ADIGE OVEST", "comune": "VERONA (VR)"},
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
# OPERA DA PROCESSARE
# --------------------------------------------------

opera_report = "P005_Adige_Ovest"  # <-- scegli qui

# Controllo coerenza
if opera_report not in opere:
    raise ValueError(
        f"Opera '{opera_report}' non presente nel dizionario opere "
        f"Disponibili: {list(opere.keys())}"
    )


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
# MAIN LOOP
# --------------------------------------------------
opera_label = opere[opera_report]["label"]
opera_comune = opere[opera_report]["comune"]

current = start_period

while current < end_period:

    year = current.year
    month = current.month

    mese_tag = f"{year}_{month:02d}"
    mese_nome = f"{MESI_IT[month]} {year}"

    start_month = current
    end_month = current + relativedelta(months=1)


    # --------------------------------------------------
    # Paths
    # --------------------------------------------------

    base_out = os.path.join("outputs", opera_report, mese_tag)
    fig_dir = os.path.join("figures", opera_report, mese_tag)

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
    # metrics_df = pd.read_csv(os.path.join(base_out, "model_metrics.csv"))

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
                if f.lower().endswith((".png", ".jpg", ".jpeg")) and 'torte' not in f
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
    # Section 1: Introduction
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
    # Section 2: Data Availability
    # --------------------------------------------------

    master.add_heading("2. Data Availability", level=1)

    # Cartella dove sono salvate le figure
    fig_dir = os.path.join("figures", opera_report, f"{year}_{month:02d}")

    # Cerca il file che contiene "torte"
    torte_file = None
    if os.path.exists(fig_dir):
        for f in os.listdir(fig_dir):
            if "torte" in f and f.lower().endswith(".png"):
                torte_file = os.path.join(fig_dir, f)
                break

    if torte_file:
        master.add_paragraph()  # spazio
        master.add_picture(torte_file, width=Inches(6.5))
        # master.add_paragraph("Figure 2.1 – Data availability summary (control limits ±3σ).")
    else:
        master.add_paragraph("Torte plot not available for this period.")

    master.add_page_break()


    # --------------------------------------------------
    # Section 3: Raw Data Visualization
    # --------------------------------------------------

    master.add_heading("3. Raw Data", level=1)

    master.add_paragraph(
        "Si riportano di seguito i grafici dei dati grezzi suddivisi per tipologia "
        "di sensore, relativi al periodo di riferimento di questo report.\n"
        "All'andamento di ogni sensore è affiancato l'andamento della temperatura"
        "di una sonda di riferimento, per evidenziare eventuali correlazioni.\n"
    )

    # Dizionario tipologie
    tipologie_raw = {
        "Inclinometri": "raw_ICD",
        "Potenziometri": "raw_POT",
        "Estensimetri": "raw_EST",
    }

    for nome_tipologia, prefisso in tipologie_raw.items():

        # Filtra grafici della tipologia
        tipo_figures = sorted([f for f in figures if f.startswith(prefisso)])

        # Se non esistono grafici → salta completamente
        if not tipo_figures:
            continue

        # Titolo livello 2
        master.add_heading(nome_tipologia, level=2)

        # Tabella 2 colonne per immagini
        table = master.add_table(rows=1, cols=2)

        for i in range(0, len(tipo_figures), 2):
            row_cells = table.add_row().cells

            row_cells[0].paragraphs[0].add_run().add_picture(
                os.path.join(fig_dir, tipo_figures[i]), width=Cm(9)
            )

            if i < len(tipo_figures) - 1:
                row_cells[1].paragraphs[0].add_run().add_picture(
                    os.path.join(fig_dir, tipo_figures[i + 1]), width=Cm(9)
                )

        master.add_paragraph()  # spazio dopo ogni blocco

    # Se non esiste nessun raw grafico in generale
    if not any(f.startswith("raw_") for f in figures):
        master.add_paragraph("Non sono disponibili grafici dei sensori per il mese selezionato.")

    master.add_page_break()


    # --------------------------------------------------
    # Section 4: Z-Score Visualization
    # --------------------------------------------------

    master.add_heading("4. Z-Score Visualization", level=1)

    master.add_paragraph(
        "Si riportano di seguito i grafici degli z-score suddivisi per tipologia "
        "di sensore, relativi al periodo di riferimento di questo report.\n"
        "Per z-score si intende il numero di deviazioni standard con cui "
        "una misura si discosta dalla relativa media.\n"
    )


    # Funzione per classificare in base al suffisso
    def classifica_tipologia(nome_file):
        if not nome_file.startswith("z-score"):
            return None

        base = os.path.splitext(nome_file)[0]  # rimuove .png
        suffisso = base[-1].lower()

        if suffisso in ["x", "y"]:
            return "Inclinometri"
        elif suffisso == "s":
            return "Potenziometri"
        elif suffisso == "e":
            return "Estensimetri"
        else:
            return None


    # Costruisco dizionario dinamico
    tipologie_zscore = {
        "Inclinometri": [],
        "Potenziometri": [],
        "Estensimetri": []
    }

    for f in figures:
        categoria = classifica_tipologia(f)
        if categoria:
            tipologie_zscore[categoria].append(f)


    for nome_tipologia, tipo_figures in tipologie_zscore.items():

        tipo_figures = sorted(tipo_figures)

        if not tipo_figures:
            continue

        master.add_heading(nome_tipologia, level=2)

        table = master.add_table(rows=1, cols=2)

        for i in range(0, len(tipo_figures), 2):
            row_cells = table.add_row().cells

            row_cells[0].paragraphs[0].add_run().add_picture(
                os.path.join(fig_dir, tipo_figures[i]), width=Cm(9)
            )

            if i < len(tipo_figures) - 1:
                row_cells[1].paragraphs[0].add_run().add_picture(
                    os.path.join(fig_dir, tipo_figures[i + 1]), width=Cm(9)
                )

        master.add_paragraph()

    master.add_page_break()


    # --------------------------------------------------
    # Section 5: Warnings and Alerts
    # --------------------------------------------------

    master.add_heading("5. Warnings and Alerts", level=1)

    # larghezze
    col_widths = [Cm(2), Cm(2), Cm(2)]

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
    # Section 4: Model Performance
    # --------------------------------------------------

    # master.add_heading("4. Model Performance", level=1)

    # table = master.add_table(rows=1, cols=len(metrics_df.columns))

    # for i, col in enumerate(metrics_df.columns):
    #     table.rows[0].cells[i].text = col

    # for _, row in metrics_df.iterrows():
    #     row_cells = table.add_row().cells
    #     for i, val in enumerate(row):
    #         row_cells[i].text = f"{val:.3f}"


    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_name = f"{cliente}_{opera_report}_{mese_tag}.docx"
    output_dir = os.path.join("outputs", opera_report, mese_tag)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_name)
    master.save(output_path)

    master.save(output_path)
    print(f"\033[92m✔ salvato {output_name}\033[0m")

    current += relativedelta(months=1)
