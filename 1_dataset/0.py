import csv
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir) 
file_input = os.path.join(PROJECT_ROOT, "csv", "complete_series.csv")
file_output = os.path.join(PROJECT_ROOT, "csv", "complete_series_corretto.csv")

serie_viste = set()
righe_cancellate = 0  # Contatore per le righe rimosse

with open(file_input, mode='r', encoding='utf-8') as f_in, \
     open(file_output, mode='w', encoding='utf-8', newline='') as f_out:
    
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
    writer.writeheader()
    
    for row in reader:
        nome = row['Nome_Serie']
        
        if nome not in serie_viste:
            writer.writerow(row)
            serie_viste.add(nome)
        else:
            # Se la serie è già stata vista, incrementiamo il contatore e la saltiamo
            righe_cancellate += 1

print(f"Operazione completata! Sono state cancellate esattamente {righe_cancellate} righe duplicate.")