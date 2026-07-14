# da eseguire in caso di interruzione esecuzione di 5_img_730_giorni.py o 8_image_balance.py per mantenere consistenza csv
# se si trovano serie complete gli script aggiungono subito la riga in complete_series.csv
# rimuovono righe da incomplete_series.csv solo al termine dell'esecuzione
# se interrompo ho aggiunto righe a complete senza toglierle da incomplete
#
# rimuove doppioni in complete e cancella serie da incomplete presenti già in complete
#
# INPUT:
#   complete_series.csv
#   incomplete_series.csv
#
# OUTPUT:
#   complete_series.csv (modificato)
#   incomplete_series.csv (modificato)


import pandas as pd

# rimuove doppioni dal file dei completi
df_comp = pd.read_csv("csv/new_complete_series_gee_COMPLETO.csv")
df_comp.drop_duplicates(subset=['Nome_Serie'], inplace=True)
df_comp.to_csv("csv/new_complete_series_gee_COMPLETO.csv", index=False)

# rimuove dal file degli incompleti le serie che sono già nei completi
df_inc = pd.read_csv("csv/new_incomplete_series_gee_COMPLETO.csv")
df_inc = df_inc[~df_inc['Nome_Serie'].isin(df_comp['Nome_Serie'])]
df_inc.to_csv("csv/new_incomplete_series_gee_COMPLETO.csv", index=False)