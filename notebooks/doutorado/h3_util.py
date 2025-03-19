import h3
import pandas as pd

def load_bairro_data(h3_index_file):
    """
    Carrega os dados dos bairros a partir de um arquivo, onde a coluna 'h3_index10'
    contém uma lista de índices H3.

    Retorna: um dicionário onde a chave é o índice H3 e o valor é o nome do bairro.
    """
    df = pd.read_csv(h3_index_file)
    bairro_map = {}
    for index, row in df.iterrows():
        h3_index_list = row['h3_index10']
        bairro = row['name']

        # Verifique se h3_index_list é realmente uma lista (após ler do CSV, pode ser uma string)
        if isinstance(h3_index_list, str):
            # Tenta converter a string em uma lista (assumindo que está formatada como uma lista Python)
            try:
                h3_index_list = eval(h3_index_list)  # Use eval() com CUIDADO! Veja a nota abaixo.
            except (SyntaxError, NameError):
                print(f"Erro ao converter string para lista na linha {index}. Pulando.")
                continue  # Pula para a próxima linha do DataFrame

        if isinstance(h3_index_list, list):  # Garante que agora é uma lista
            for h3_index in h3_index_list:
                bairro_map[h3_index] = bairro
        else:
            print(f"Valor inesperado na coluna 'h3_index10' na linha {index}. Esperava uma lista. Pulando.")
            continue # Pula para a próxima linha do DataFrame

    return bairro_map

def get_bairro_from_h3(h3_index, bairro_map, name_norm):
    """
    Determina o bairro ao qual um índice H3 pertence, usando o mapa pré-carregado.
    """
    if h3_index in bairro_map:
        return bairro_map[h3_index]
    else:
        try:
            bairro = name_norm.upper().split('-')[-1].strip()
            if bairro in set(bairro_map.values()):
                return bairro
            else:
                return 'REGIÃO METROPOLITANA'
        except:
            return 'BAIRRO DESCONHECIDO'

def lat_lng_to_h3(latitude, longitude, resolution):
  """
  Converte uma coordenada de latitude e longitude em um índice H3.

  Args:
    latitude: A latitude da coordenada (em graus decimais).
    longitude: A longitude da coordenada (em graus decimais).
    resolution: A resolução H3 desejada (um inteiro entre 0 e 15).

  Returns:
    O índice H3 correspondente à coordenada na resolução especificada (uma string).
  """
  h3_index = h3.latlng_to_cell(latitude, longitude, resolution)
  return h3_index