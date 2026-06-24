import csv
import os


# File di input 
org_s1_train = "/workspace/csv/train/org/S1RTC_train.csv"
org_s2_train_1 = "/workspace/csv/train/org/S2L2A_train_1.csv"
org_s2_train_2 = "/workspace/csv/train/org/S2L2A_train_2.csv"
org_s2_train_3 = "/workspace/csv/train/org/S2L2A_train_3.csv"

org_s1_val = "/workspace/csv/val/org/S1RTC_val.csv"
org_s2_val = "/workspace/csv/val/org/S2L2A_val.csv"

org_s1_test = "/workspace/csv/test/org/S1RTC_test.csv"
org_s2_test = "/workspace/csv/test/org/S2L2A_test.csv"

# file csv output  
clean_s1_train = "/workspace/csv/train/clean/S1RTC_train.csv"
clean_s2_train_1 = "/workspace/csv/train/clean/S2L2A_train_1.csv"
clean_s2_train_2 = "/workspace/csv/train/clean/S2L2A_train_2.csv"
clean_s2_train_3 = "/workspace/csv/train/clean/S2L2A_train_3.csv"

clean_s1_val = "/workspace/csv/val/clean/S1RTC_val.csv"
clean_s2_val = "/workspace/csv/val/clean/S2L2A_val.csv"

clean_s1_test = "/workspace/csv/test/clean/S1RTC_test.csv"
clean_s2_test = "/workspace/csv/test/clean/S2L2A_test.csv"

def pulisci_csv(percorso_input, percorso_output):
    
    if not os.path.exists(percorso_input):
        print(f"❌ Il file {percorso_input} non esiste.")
        return

    # caratteri non validi
    valori_non_validi = ["", "n/a", "nan", "none", "null"]

    serie_totali = 0
    serie_valide = 0

    print(f"Esecuzione: {os.path.basename(percorso_input)}...")

    with open(percorso_input, mode='r', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)
        
        # intestazione
        intestazione = next(reader, None)
        if intestazione is None:
            print("🟡 File vuoto")
            return

        with open(percorso_output, mode='w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(intestazione) 

            # esecuzione
            for riga in reader:
                serie_totali += 1
                serie_da_salvare = True
                
                for cella in riga:
                    valore_pulito = cella.strip().lower() 
                    
                    if valore_pulito in valori_non_validi:
                        serie_da_salvare = False
                        break 
                
                # salva se riga è true
                if serie_da_salvare:
                    writer.writerow(riga)
                    serie_valide += 1

    # calcolo statistiche
    serie_rimosse = serie_totali - serie_valide
    
    print(f"🔵 File pulito salvato in: {os.path.basename(percorso_output)}")
    print(f"Serie originali:  {serie_totali}")
    print(f"Serie scartate:  {serie_rimosse}")
    print(f"Serie rimanenti: {serie_valide}\n")

# main

print(f"---------- TRAIN SET ----------")
pulisci_csv(org_s1_train, clean_s1_train)
pulisci_csv(org_s2_train_1, clean_s2_train_1)
pulisci_csv(org_s2_train_2, clean_s2_train_2)
pulisci_csv(org_s2_train_3, clean_s2_train_3)

print(f"\n---------- VAL SET ----------")
pulisci_csv(org_s1_val, clean_s1_val)
pulisci_csv(org_s2_val, clean_s2_val)

print(f"\n---------- TEST SET ----------")
pulisci_csv(org_s1_test, clean_s1_test)
pulisci_csv(org_s2_test, clean_s2_test)
print("\n✔️ FINITO")





