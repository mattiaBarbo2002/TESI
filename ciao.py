"""

print(f"  _____________________________________")
print(f"#                                       #")
print(f"||                __        __         ||")
print(f"||      /\\  |  | | _  |  | |__| |      ||")
print(f"||     /--\\ |__| |__| |__| | \\  |      ||")
print(F"||                                     ||")
print(f"||            __  __       __          ||")
print(f"||    |\\ | | |   |  | |   |__   ||     ||")
print(f"||    | \\| | |__ |__| |__ |__  \\__/    ||")
print(f"||                                     ||")
print(f"# _____________________________________ #")

"""
















import time

print()
print()
print()



# Mettiamo ogni riga del tuo disegno dentro una lista.
# Python tratterà ogni stringa come se fosse una riga della nostra griglia!
disegno = [
    "🎂🎉🎊🥳🎁🎈🎂🎉🎊🥳🎁🎈🎂🎉🎊🥳🎁🎈🎂🎉",
    "🎈                __        __        🎊",
    "🎁      /\\  |  | | _  |  | |__| |     🥳",
    "🥳     /--\\ |__| |__| |__| | \\  |     🎁",
    "🎊                                    🎈",
    "🎉            __  __       __         🎂",
    "🎂    |\\ | | |   |  | |   |__   ||    🎉",
    "🎈    | \\| | |__ |__| |__ |__  \\__/   🎊",
    "🎁                                    🥳",
    "🥳🎊🎉🎂🎈🎁🥳🎊🎉🎂🎈🎁🥳🎊🎉🎂🎈🎁🥳🎊"
]

# len(disegno) conta in automatico quante righe ci sono (11)
for riga in range(len(disegno)):
    
    # len(disegno[riga]) conta quanti caratteri ci sono nella riga attuale
    for colonna in range(len(disegno[riga])):
        
        # Peschiamo il singolo carattere (spazio, stanghetta, lettera)
        simbolo = disegno[riga][colonna]
        
        print(simbolo, end="", flush=True)
        time.sleep(0.04)  # Pausa di 1 centesimo di secondo
        
    # Andiamo a capo alla fine di ogni riga del disegno
    print()

print()
print()
print()
print()
print()
print()













