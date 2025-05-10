#!/usr/bin/env python3

"""
Este script modifica el archivo Sonitranslate_openai.ipynb para agregar 
la instalación de google-generativeai.
"""

import json
import os

def add_gemini_install():
    notebook_path = 'Sonitranslate_openai.ipynb'
    
    try:
        # Leer el notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Buscar la celda que contiene la instalación de paquetes
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and 'pip install --upgrade openai' in ''.join(cell['source']):
                # Encontramos la celda de instalación, vamos a agregar google-generativeai
                source_lines = cell['source']
                
                # Buscar la última línea de instalación
                for j, line in enumerate(reversed(source_lines)):
                    if 'pip install' in line or '!sudo apt' in line:
                        # Insertar después de la última instalación
                        gemini_install = [
                            '\nprint("Instalando Google Generative AI...")\n',
                            '!pip install google-generativeai\n'
                        ]
                        
                        # Calcular el índice real (teniendo en cuenta que estamos recorriendo la lista en reversa)
                        real_index = len(source_lines) - 1 - j
                        
                        # Insertar las nuevas líneas después de la última instalación
                        new_source = source_lines[:real_index+1] + gemini_install + source_lines[real_index+1:]
                        notebook['cells'][i]['source'] = new_source
                        break
                
                break
        
        # Guardar el notebook modificado
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)
        
        print(f"El archivo {notebook_path} ha sido modificado exitosamente.")
        print("Se ha agregado la instalación de google-generativeai.")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    add_gemini_install() 