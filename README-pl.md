# Active Learning MedMNIST

Repozytorium zawiera kod źródłowy, pliki konfiguracyjne oraz wyniki przeprowadzonych badań nad zastosowaniem **aktywnego uczenia** w klasyfikacji obrazów medycznych z kolekcji **MedMNIST2D**.

Celem projektu jest porównanie aktywnego uczenia z klasycznym uczeniem nadzorowanym. W aktywnym uczeniu model nie korzysta od początku z całego oznaczonego zbioru treningowego, lecz stopniowo wybiera próbki, które mają zostać oznaczone i włączone do dalszego treningu. Eksperymenty służą sprawdzeniu, czy taki sposób doboru danych pozwala osiągnąć jakość klasyfikacji zbliżoną do uczenia na pełnym zbiorze przy wykorzystaniu mniejszej liczby etykiet.

## Zakres projektu

Projekt obejmuje:

- przygotowanie danych MedMNIST2D w formacie `.npy`;
- przekształcenie wybranych zbiorów w zadania klasyfikacji binarnej;
- trening modelu ResNet-18 w schemacie uczenia nadzorowanego;
- trening z wykorzystaniem aktywnego uczenia i różnych strategii wyboru próbek;
- automatyczne uruchamianie serii eksperymentów dla wielu konfiguracji;
- zapisywanie wyników w plikach CSV;
- zapisywanie stanów wytrenowanych modeli;
- zapisywanie logów z wyboru próbek w kolejnych cyklach aktywnego uczenia;
- analizę wyników uczenia nadzorowanego;
- wstępny dobór hiperparametrów aktywnego uczenia oraz końcowe porównanie strategii;
- analizę wpływu optymalizacji progu decyzyjnego.

## Dane

W eksperymentach wykorzystywane są wybrane zbiory z kolekcji MedMNIST2D:

- `bloodmnist` - obrazy mikroskopowe komórek krwi;
- `octmnist` - obrazy OCT siatkówki oka;
- `pathmnist` - obrazy histopatologiczne tkanki jelita grubego;
- `pneumoniamnist` - obrazy RTG klatki piersiowej.

Zbiór BloodMNIST wykorzystano do wstępnego doboru hiperparametrów aktywnego uczenia. Pozostałe trzy zbiory posłużyły do końcowej oceny i porównania klasyfikatorów trenowanych w schemacie uczenia nadzorowanego i aktywnego.

Każdy przygotowany zbiór powinien mieć następującą strukturę:

```text
data/<dataset>/
  train_images.npy
  train_labels.npy
  val_images.npy
  val_labels.npy
  test_images.npy
  test_labels.npy
```

Sposób pobierania i przygotowania danych opisano w pliku
[`data/readme.md`](data/readme.md).

## Model

W eksperymentach stosowany jest model oparty na architekturze ResNet-18 dostępnej w bibliotece `torchvision`.

Model korzysta z wag wytrenowanych wstępnie na zbiorze ImageNet. W przypadku obrazów RGB pierwsza warstwa konwolucyjna pozostaje zgodna z modelem źródłowym. Dla obrazów jednokanałowych warstwa ta jest dostosowywana do liczby kanałów wejściowych i inicjalizowana od nowa, natomiast pozostałe bloki ResNet-18 nadal korzystają z wag wytrenowanych na ImageNet.

Końcowa warstwa klasyfikacyjna jest zastępowana warstwą odpowiadającą liczbie klas w danym zadaniu. Bezpośrednio przed nią znajduje się warstwa dropout, wykorzystywana między innymi przez strategie oparte na MC Dropout.

## Strategie aktywnego uczenia

Projekt obsługuje następujące strategie wyboru próbek:

- `random` - losowy wybór próbek, pełniący funkcję punktu odniesienia;
- `least_confident` - wybór próbek, dla których najwyższe przewidywane prawdopodobieństwo klasy jest najmniejsze;
- `margin` - wybór próbek z najmniejszą różnicą między dwoma najwyższymi prawdopodobieństwami klas;
- `entropy` - wybór próbek o największej entropii predykcji;
- `mc_entropy` - wariant oparty na entropii obliczanej z wielu stochastycznych predykcji uzyskanych za pomocą MC Dropout;
- `mc_bald` - strategia BALD wykorzystująca MC Dropout do oceny niepewności epistemicznej modelu;
- `entropy_diverse` - połączenie wyboru próbek niepewnych z doborem uwzględniającym ich zróżnicowanie w przestrzeni reprezentacji cech;
- `mc_entropy_diverse` - połączenie MC Entropy z doborem zróżnicowanej partii próbek;
- `mc_bald_diverse` - połączenie BALD z doborem zróżnicowanej partii próbek;
- `egl_fc` - strategia Expected Gradient Length obliczana dla ostatniej warstwy w pełni połączonej.

Zestaw strategii używanych w konkretnym eksperymencie określa się w odpowiednim pliku YAML w katalogu `configs/`.

## Metryki

Wyniki eksperymentów opisywane są przede wszystkim za pomocą następujących metryk:

- `acc` - dokładność klasyfikacji;
- `f1_macro` - miara F1 obliczana w wariancie makro;
- `auc` - pole pod krzywą ROC;
- `ap` - średnia precyzja;
- `val_mean` - średnia z metryk obliczonych na zbiorze walidacyjnym.

Dla aktywnego uczenia zapisywane są ponadto liczba aktualnie oznaczonych próbek (`labeled_count`) oraz parametry opisujące budżet anotacji.

## Instalacja

Projekt najlepiej uruchamiać w wirtualnym środowisku Pythona:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Po aktywowaniu środowiska wszystkie skrypty można uruchamiać jako moduły Pythona z głównego katalogu repozytorium.

## Przykładowe polecenia

Pojedynczy eksperyment aktywnego uczenia:

```bash
python3 -m src.training.train_active -c configs/train_active/default.yaml
```

Pojedynczy trening nadzorowany:

```bash
python3 -m src.training.train_supervised -c configs/train_supervised/default.yaml
```

Seria eksperymentów aktywnego uczenia:

```bash
python3 -m src.run_experiments -c configs/run_experiments/active/default.yaml
```

Seria eksperymentów uczenia nadzorowanego:

```bash
python3 -m src.run_experiments -c configs/run_experiments/supervised/default.yaml
```

Analiza wyników doboru hiperparametrów aktywnego uczenia:

```bash
python3 -m src.analysis.analyze_active_screening -c configs/analysis/analyze_active_screening/default.yaml
```

Analiza wyników uczenia nadzorowanego:

```bash
python3 -m src.analysis.analyze_supervised -c configs/analysis/analyze_supervised/default.yaml
```

Analiza końcowego porównania strategii aktywnego uczenia:

```bash
python3 -m src.analysis.analyze_active_strategies -c configs/analysis/analyze_active_strategies/default.yaml
```

Analiza wpływu progu decyzyjnego:

```bash
python3 -m src.analysis.analyze_threshold_optimization -c configs/analysis/analyze_threshold_optimization/default.yaml
```

## Typowy przebieg eksperymentu

1. Aktywuj środowisko `venv`.
2. Zainstaluj zależności wymienione w pliku `requirements.txt`.
3. Upewnij się, że dane znajdują się w katalogu `data/<dataset>/`.
4. Wybierz odpowiedni plik konfiguracyjny YAML z katalogu `configs/` albo utwórz jego kopię.
5. Uruchom pojedynczy trening lub serię eksperymentów za pomocą modułu `src.run_experiments`.
6. Sprawdź zapisane wyniki w katalogu `results/`.
7. W przypadku aktywnego uczenia sprawdź również stany modeli w katalogu `models/` i logi wyboru próbek w katalogu `logs/`.
8. Uruchom odpowiedni skrypt analityczny, aby opracować wyniki treningu nadzorowanego, doboru hiperparametrów, porównania strategii lub optymalizacji progu decyzyjnego.

## Struktura repozytorium

```text
configs/
data/
logs/
models/
results/
src/
requirements.txt
README.md
README-pl.md
```

### `configs/`

Pliki YAML określające parametry uruchamiania skryptów. Dla każdego głównego rodzaju eksperymentu dostępny jest plik `default.yaml`, który pełni funkcję przykładowej konfiguracji i dokumentacji dostępnych ustawień.

Więcej informacji: [`configs/readme.md`](configs/readme.md).

### `data/`

Dane wejściowe w formacie `.npy`, podzielone na zbiory treningowy (`train`), walidacyjny (`val`) i testowy (`test`). Repozytorium może zawierać tylko część danych. Pozostałe zbiory należy pobrać zgodnie z dołączoną instrukcją.

Więcej informacji: [`data/readme.md`](data/readme.md).

### `logs/`

Dzienniki wyboru próbek tworzone podczas aktywnego uczenia. Pliki `.asklog.json` zawierają identyfikatory próbek wybranych do oznaczenia w kolejnych cyklach.

W repozytorium znajduje się reprezentatywny przykład dziennika dla eksperymentu `final_active_best/octmnist`. Pozostałe dzienniki są generowane podczas uruchamiania eksperymentów i domyślnie nie podlegają wersjonowaniu.

Więcej informacji: [`logs/readme.md`](logs/readme.md).

### `models/`

Stany modeli zapisane w plikach `.pth`. Ze względu na ich rozmiar większość z nich nie podlega wersjonowaniu. W repozytorium pozostawiono jeden przykładowy plik tego rodzaju.

Więcej informacji: [`models/readme.md`](models/readme.md).

### `results/`

Pliki CSV z wynikami eksperymentów oraz katalogi zawierające utworzone na ich podstawie zestawienia, tabele i wykresy. Końcowe pliki `results/final_*.csv` są wersjonowane, natomiast pozostałe pliki wynikowe są domyślnie pomijane.

Więcej informacji: [`results/readme.md`](results/readme.md).

### `src/`

Kod źródłowy projektu. Obejmuje skrypty treningowe i analityczne, mechanizm uruchamiania serii eksperymentów, funkcje tworzące wykresy oraz narzędzia pomocnicze odpowiedzialne za wczytywanie danych, mapowanie etykiet i obsługę modelu.

Więcej informacji: [`src/readme.md`](src/readme.md).

## Najważniejsze pliki źródłowe

- `src/training/train_supervised.py` - trening i ewaluacja modelu w schemacie uczenia nadzorowanego;
- `src/training/train_active.py` - przeprowadzenie pojedynczego eksperymentu aktywnego uczenia;
- `src/run_experiments.py` - uruchamianie serii eksperymentów dla wielu konfiguracji;
- `src/analysis/analyze_supervised.py` - analiza wyników uczenia nadzorowanego;
- `src/analysis/analyze_active_screening.py` - analiza wyników doboru hiperparametrów aktywnego uczenia;
- `src/analysis/analyze_active_strategies.py` - analiza końcowego porównania strategii aktywnego uczenia;
- `src/analysis/analyze_threshold_optimization.py` - analiza wpływu optymalizacji progu decyzyjnego;
- `src/analysis/plotting.py` - funkcje tworzące wykresy;
- `src/utils/load_data.py` - wczytywanie i przygotowanie danych;
- `src/utils/labels_mapping.py` - przekształcanie etykiet na potrzeby klasyfikacji binarnej;
- `src/utils/shared.py` - wspólne elementy implementacji, w tym model ResNet-18, obliczanie predykcji i metryk oraz zapis wyników w plikach CSV.

## Wyniki końcowe

W repozytorium znajdują się trzy podstawowe pliki zawierające wyniki eksperymentów:

```text
results/final_supervised.csv
results/final_active_screening.csv
results/final_active_best.csv
```

Poszczególne pliki zawierają:

- `final_supervised.csv` - wyniki uczenia nadzorowanego stanowiące punkt odniesienia dla dalszych porównań;
- `final_active_screening.csv` - wyniki wstępnego doboru hiperparametrów aktywnego uczenia na zbiorze `bloodmnist`;
- `final_active_best.csv` - wyniki końcowego porównania strategii aktywnego uczenia dla wybranej konfiguracji.

Na podstawie tych plików skrypty analityczne tworzą między innymi podsumowania metryk testowych, zestawienia dotyczące doboru hiperparametrów, rankingi strategii, analizę wpływu progu decyzyjnego oraz zagregowane macierze pomyłek.

## Odtwarzalność eksperymentów

Skrypty ustawiają ziarna generatorów liczb pseudolosowych używanych przez Pythona, NumPy i PyTorch. Nie gwarantuje to jednak uzyskania identycznych wyników w każdym środowisku. Na rezultat mogą wpływać między innymi wersje bibliotek, użyte urządzenie obliczeniowe, sterowniki CUDA oraz pozostałe elementy konfiguracji systemu.

Najważniejsze parametry eksperymentów zapisano w plikach YAML oraz w wynikowych plikach CSV.
