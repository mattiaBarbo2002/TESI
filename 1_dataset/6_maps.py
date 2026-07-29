import re
from pyproj import Transformer

def ottieni_link_google_maps(nome_serie):
    
    match = re.search(r'_(\d{1,2})([C-X])[A-Z]*_x(\d+)_y(\d+)', nome_serie, re.IGNORECASE)
    
    if not match:
        
        print("❌ Formato del nome non riconosciuto. Verifica la stringa!")
        return None
        
    zona_utm = int(match.group(1))       
    lettera_fascia = match.group(2).upper()  
    x_easting = float(match.group(3))    
    y_northing = float(match.group(4))   
    
    if lettera_fascia >= 'N':
        epsg_partenza = 32600 + zona_utm  
        emisfero = "Nord"
    else:
        epsg_partenza = 32700 + zona_utm  
        emisfero = "Sud"
        
    transformer = Transformer.from_crs(f"EPSG:{epsg_partenza}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x_easting, y_northing)
    
    link_maps = f"https://www.google.com/maps?q={lat},{lon}"
    
    print(f"SERIE:      {nome_serie}")
    print(f"UTM:        {zona_utm}{lettera_fascia} (Emisfero {emisfero})")
    print(f"Coordinate: {lat:.5f}, {lon:.5f}")
    print(f"Maps:       {link_maps}")
    
    return link_maps

if __name__ == "__main__":
    
    serie = "EMSR557_13_32VNP_x513675_y6856995"
    ottieni_link_google_maps(serie)