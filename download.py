from ftplib import FTP, error_perm
import os
import re 
from datetime import datetime, date
import pandas as pd

host = "172.29.3.38"
user = "csv"
password = "csv"
base_remote = "/ALL/"
base_local = r"C:/Users/m.zortea/Desktop/gua"
start_date = date(2025, 1, 1)
end_date = date(2025, 10, 1)

def is_directory(ftp, name):
    """Ritorna True se 'name' è una directory sull'FTP."""
    current = ftp.pwd()
    try:
        ftp.cwd(name)
        ftp.cwd(current)
        return True
    except error_perm:
        return False

def scarica_contenuto(ftp, remote_dir, local_dir):
    """Scarica ricorsivamente tutto da remote_dir, saltando 'csv_acc'."""
    ftp.cwd(remote_dir)
    os.makedirs(local_dir, exist_ok=True)
    elementi = ftp.nlst()
    print(f"Contenuto di {remote_dir}: {elementi}")
    
    for elemento in elementi:
        if elemento in (".", ".."):
            continue

        # Costruisci path completi

        remote_path = f"{remote_dir}/{elemento}"
        local_path = os.path.join(local_dir, elemento)
        local_path_csv = os.path.join(local_dir, elemento)
        local_path_parquet = os.path.splitext(local_path_csv)[0] + ".parquet"

        # Verifica se è directory o file
        if is_directory(ftp, elemento):
            if elemento in ["csv_acc"]:
                continue
            scarica_contenuto(ftp, remote_path, local_path)

        else:
            print(f"Download File: {remote_path}")
            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR " + elemento, f.write)

            # Leggi CSV e salva Parquet
            try:
                df = pd.read_csv(local_path_csv, delimiter=';')
                df.to_parquet(local_path_parquet, index=False)
            except Exception as e:
                print(f"Errore nella conversione di {elemento}: {e}")

            # Opzionale: rimuovi il CSV originale se vuoi solo Parquet
            os.remove(local_path_csv)

    ftp.cwd("..")


def main():

    os.makedirs(base_local, exist_ok=True)
    

    ftp = FTP(host)
    ret = ftp.login(user, password)

    print(ret)

    ftp.cwd(base_remote)

    all_folders = sorted(ftp.nlst(), reverse=True)
    
    for el in all_folders[:]:
        date = datetime.strptime(el[:8], "%Y%m%d").date()
        if date < start_date or date > end_date:
            all_folders.remove(el)
        else:
            print(f"\nScaricamento cartella: {el}")
            scarica_contenuto(ftp, os.path.join(base_remote, el), base_local)
    ftp.quit()
    print("\nDownload completato")


if __name__ == "__main__":
    main()
