#!/usr/bin/env python3

"""
Este script modifica el archivo Sonitranslate_openai.ipynb para agregar 
soporte para Google Gemini API en la interfaz de usuario.
"""

import json
import os

def add_gemini_support():
    notebook_path = 'Sonitranslate_openai.ipynb'
    
    try:
        # Leer el notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Buscar la celda que contiene la configuración de OpenAI API
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and '#@title Run' in cell['source'][0]:
                # Encontramos la celda correcta, ahora vamos a modificarla
                source_lines = cell['source']
                
                # Buscar la línea después de la entrada de OPENAI_API_KEY_INPUT
                for j, line in enumerate(source_lines):
                    if 'OPENAI_API_KEY_INPUT' in line:
                        # Insertar las nuevas líneas después de esta
                        gemini_lines = [
                            '#@markdown `در صورتی که کاربرد این تیک را نمیدانید ، فعال نکنید`\n',
                            'REQUIRE_GEMINI_KEY = False #@param {type:"boolean"}\n',
                            '#@markdown ## `(اختیاری) gemini محل قرار گرفتن کلید `\n',
                            'GEMINI_API_KEY_INPUT = "" #@param {type:\'string\'}\n'
                        ]
                        
                        # Buscar la línea donde se importa os
                        for k, os_line in enumerate(source_lines[j+1:], j+1):
                            if 'import os' in os_line:
                                # Encontramos la línea de import os, no la modificamos
                                os_index = k
                                break
                        
                        # Buscar el final del bloque de configuración de OpenAI API
                        for k, env_line in enumerate(source_lines[os_index+1:], os_index+1):
                            if '%env YOUR_HF_TOKEN' in env_line:
                                # Encontramos donde termina la configuración de OpenAI
                                # Agregar código de Gemini antes de esta línea
                                gemini_code = [
                                    '\n',
                                    '# تنظیم API key برای Gemini\n',
                                    'gemini_key = GEMINI_API_KEY_INPUT\n',
                                    'require_gemini_key = REQUIRE_GEMINI_KEY\n',
                                    '\n',
                                    'if gemini_key:\n',
                                    '    os.environ[\'GEMINI_API_KEY\'] = gemini_key\n',
                                    '    print("کلید Gemini API با موفقیت از ورودی تنظیم شد.")\n',
                                    'elif \'GEMINI_API_KEY\' in os.environ:\n',
                                    '    # پاک کردن کلید قبلی اگر ورودی خالی است\n',
                                    '    del os.environ[\'GEMINI_API_KEY\']\n',
                                    '    print("کلید Gemini API پاک شد (ورودی خالی بود).")\n',
                                    'else:\n',
                                    '    print("کلید Gemini API ارائه نشده است.")\n',
                                    '\n',
                                    '# بررسی اینکه آیا کلید الزامی است ولی ارائه نشده\n',
                                    'if require_gemini_key and not os.environ.get("GEMINI_API_KEY"):\n',
                                    '    raise ValueError("کادر \'نیاز به کلید Gemini\' فعال است اما کلیدی وارد نشده است. لطفاً کلید معتبر خود را وارد کنید یا تیک این کادر را بردارید.")\n'
                                ]
                                
                                # Insertar las nuevas líneas
                                new_source = source_lines[:j+1] + gemini_lines + source_lines[j+1:k] + gemini_code + source_lines[k:]
                                notebook['cells'][i]['source'] = new_source
                                break
                        
                        break
        
        # Guardar el notebook modificado
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)
        
        print(f"El archivo {notebook_path} ha sido modificado exitosamente.")
        print("Se ha agregado soporte para Gemini API.")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    add_gemini_support() 