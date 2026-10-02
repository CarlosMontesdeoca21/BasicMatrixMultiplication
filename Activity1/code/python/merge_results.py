import csv
import os
import statistics

def main():
    input_files = [
        "Activity1/data/results_c.csv",
        "Activity1/data/results_java.csv",
        "Activity1/data/results_python.csv"
    ]
    output_file = "Activity1/data/benchmark_results.csv"

    print("=== UNIFICADOR Y ANALIZADOR DE BENCHMARKS ===")

    master_rows = []
    header = None

    # Leer y combinar todos los archivos individuales
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"[AVISO] No se encontró el archivo: {file_path}. Se omitirá.")
            continue
        
        print(f"Leyendo: {file_path}")
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            file_header = next(reader, None)
            
            if header is None:
                header = file_header
            
            for row in reader:
                if row:
                    master_rows.append(row)

    if not header:
        print("[ERROR] No se encontraron datos válidos para unificar.")
        return

    # Escribir el archivo maestro consolidado
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(master_rows)

    print("---------------------------------------------")
    print(f"¡Archivo maestro creado con éxito en: {output_file}!")
    print(f"Total de registros consolidados: {len(master_rows)}")
    print("\n--- ANÁLISIS ESTADÍSTICO DE TIEMPOS (Para el informe) ---")

    # Agrupar datos por (size, language) para calcular estadísticas de tiempo
    grouped_data = {}
    for row in master_rows:
        # Estructura del CSV: size, language, numeric_type, repetition, time_seconds, memory
        size = int(row[0])
        language = row[1]
        time_val = float(row[4])

        key = (size, language)
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(time_val)

    # Calcular y mostrar mediana y variabilidad (desviación estándar / rango)
    print(f"{'Tamaño (n)':<12} | {'Lenguaje':<10} | {'Mediana (s)':<12} | {'Mín - Máx (s)':<18} | {'Desv. Típica (s)'}")
    print("-" * 75)

    for (size, language), times in sorted(grouped_data.items()):
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        
        # Desviación típica (si hay al menos 2 elementos, sino 0.0)
        stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0

        print(f"{size:<12} | {language:<10} | {median_time:<12.6f} | {min_time:.6f} - {max_time:.6f} | {stdev_time:.6f}")

    print("---------------------------------------------------------------------------")

if __name__ == "__main__":
    main()