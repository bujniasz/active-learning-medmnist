# Active Learning Medmnist

Repozytorium zawiera kod, konfiguracje i wybrane artefakty eksperymentów do pracy badawczej nad zastosowaniem **Active Learning** w klasyfikacji obrazów medycznych ze zbiorów **MedMNIST2D**.

Celem projektu jest porównanie klasycznego uczenia nadzorowanego z podejściem aktywnego uczenia, w którym model nie korzysta od razu z pełnego zbioru etykiet, tylko iteracyjnie wybiera kolejne próbki do oznaczenia. Eksperymenty sprawdzają, czy odpowiedni dobór próbek pozwala osiągnąć wyniki zbliżone do pełnego treningu przy mniejszym budżecie etykiet.

## Zakres Projektu

Projekt obejmuje:

- przygotowanie danych MedMNIST2D w formacie `.npy`;
- mapowanie wybranych zbiorów do klasyfikacji binarnej;
- trening modelu ResNet18 w trybie supervised;
- trening Active Learning z różnymi strategiami wyboru próbek;
- automatyczne uruchamianie większych siatek eksperymentów;
- zapis wyników do CSV;
- zapis checkpointów modeli;
- zapis asklogów, czyli list próbek wybranych w cyklach Active Learning;
- analizę wyników screeningu i generowanie wykresów.

## Dane

Projekt korzysta z wybranych zbiorów MedMNIST2D:

- `pneumoniamnist` - binarna klasyfikacja zapalenia płuc;
- `bloodmnist` - używany głównie do screeningu hiperparametrów Active Learning;
- `octmnist` - klasy sprowadzone do problemu binarnego;
- `pathmnist` - klasy sprowadzone do problemu binarnego, z odrzuceniem wybranych klas nieużywanych w eksperymencie.

Dane są oczekiwane w strukturze:

```text
data/<dataset>/
  train_images.npy
  train_labels.npy
  val_images.npy
  val_labels.npy
  test_images.npy
  test_labels.npy
```

Szczegóły pobierania i przygotowania danych są opisane w `data/readme.md`.

## Model

W eksperymentach używany jest model oparty o `ResNet18` z `torchvision`.

Model korzysta z wag pretrenowanych na ImageNet. Dla obrazów RGB pierwsza warstwa konwolucyjna pozostaje zgodna z modelem pretrenowanym. Dla obrazów jednokanałowych pierwsza warstwa jest dostosowywana do liczby kanałów wejściowych i inicjalizowana od nowa, natomiast dalsze bloki ResNet18 nadal korzystają z wag pretrenowanych.

Ostatnia warstwa klasyfikacyjna jest zawsze wymieniana tak, aby odpowiadała liczbie klas w danym zadaniu. Przed klasyfikatorem dodany jest dropout, wykorzystywany między innymi w strategiach MC Dropout.

## Strategie Active Learning

Projekt wspiera następujące strategie wyboru próbek:

- `random` - losowy wybór próbek, traktowany jako pasywny baseline;
- `least_confident` - wybiera próbki, dla których najwyższe prawdopodobieństwo klasy jest najniższe;
- `margin` - wybiera próbki z najmniejszą różnicą między dwiema najbardziej prawdopodobnymi klasami;
- `entropy` - wybiera próbki o najwyższej entropii predykcji;
- `mc_entropy` - wariant entropii oparty o MC Dropout, czyli wiele stochastycznych predykcji modelu;
- `mc_bald` - strategia BALD oparta o MC Dropout, mierząca niepewność epistemiczną modelu;
- `entropy_diverse` - najpierw wybiera próbki niepewne, a potem dodaje selekcję różnorodną w przestrzeni embeddingów;
- `mc_entropy_diverse` - połączenie MC Entropy z doborem różnorodnego batcha;
- `mc_bald_diverse` - połączenie BALD z doborem różnorodnego batcha;
- `egl_fc` - Expected Gradient Length liczony dla ostatniej warstwy liniowej.

Nie wszystkie strategie muszą być używane w każdym finalnym eksperymencie. Konkretne zestawy strategii są definiowane w plikach YAML w `configs/`.

## Metryki

Wyniki zapisywane są głównie przy użyciu:

- `acc` - accuracy;
- `f1_macro` - macro F1;
- `auc` - ROC AUC dla klasyfikacji binarnej;
- `ap` - average precision;
- `val_mean` - średnia z dostępnych metryk walidacyjnych.

Dla eksperymentów Active Learning zapisywana jest także liczba aktualnie oznaczonych próbek (`labeled_count`) oraz parametry budżetu etykietowania.

## Instalacja

Zalecane jest uruchamianie projektu w wirtualnym środowisku.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Po aktywacji środowiska wszystkie skrypty można uruchamiać jako moduły Pythona z głównego katalogu repozytorium.

## Przykładowe Uruchomienia

Pojedynczy eksperyment Active Learning:

```bash
python3 -m src.training.train_active -c configs/train_active/default.yaml
```

Pojedynczy trening supervised:

```bash
python3 -m src.training.train_supervised -c configs/train_supervised/default.yaml
```

Siatka eksperymentów Active Learning:

```bash
python3 -m src.run_experiments -c configs/run_experiments/active/default.yaml
```

Siatka eksperymentów supervised:

```bash
python3 -m src.run_experiments -c configs/run_experiments/supervised/default.yaml
```

Analiza wyników screeningu Active Learning:

```bash
python3 -m src.analysis.analyze_active_screening -c configs/analyze_active_screening/default.yaml
```

## Typowy Flow Pracy

1. Aktywuj środowisko `venv`.
2. Zainstaluj zależności z `requirements.txt`.
3. Upewnij się, że dane znajdują się w `data/<dataset>/`.
4. Wybierz lub skopiuj odpowiedni plik YAML z `configs/`.
5. Uruchom pojedynczy trening albo sweep przez `src.run_experiments`.
6. Sprawdź wyniki w `results/`.
7. Dla Active Learning sprawdź dodatkowo checkpointy w `models/` i asklogi w `logs/`.
8. W razie potrzeby uruchom analizę screeningu lub generowanie wykresów.

## Struktura Repozytorium

```text
  configs/
  data/
  logs/
  models/
  results/
  src/
  requirements.txt
  README.md
```

### `configs/`

Pliki konfiguracyjne YAML dla głównych skryptów. Każdy główny workflow ma swój `default.yaml`, który służy zarówno jako przykład uruchomienia, jak i dokumentacja parametrów.

Więcej: `configs/readme.md`.

### `data/`

Dane wejściowe w formacie `.npy`, podzielone na `train`, `val` i `test`. W repozytorium może znajdować się tylko część danych, a pozostałe zbiory należy pobrać zgodnie z instrukcją.

Więcej: `data/readme.md`.

### `logs/`

Asklogi generowane podczas Active Learning. Pliki `.asklog.json` zapisują, które próbki zostały wybrane do oznaczenia w kolejnych cyklach.

W repozytorium trzymany jest reprezentatywny przykład asklogów dla `final_active_best/octmnist`, a pozostałe logi są traktowane jako artefakty generowane.

Więcej: `logs/readme.md`.

### `models/`

Checkpointy modeli `.pth` zapisywane podczas treningu. Modele są duże, dlatego domyślnie nie są wersjonowane. W repozytorium znajduje się jeden reprezentatywny checkpoint przykładowy.

Więcej: `models/readme.md`.

### `results/`

CSV z wynikami eksperymentów oraz foldery z wykresami i tabelami generowanymi na podstawie tych CSV. Finalne pliki `results/final_*.csv` są wersjonowane, a pozostałe artefakty są domyślnie ignorowane.

Więcej: `results/readme.md`.

### `src/`

Kod źródłowy projektu. Zawiera skrypty treningowe, launcher eksperymentów, analizę wyników, wykresy oraz funkcje pomocnicze do ładowania danych, mapowania etykiet i obsługi modelu.

Więcej: `src/readme.md`.

## Najważniejsze Pliki Źródłowe

- `src/training/train_supervised.py` - trening i ewaluacja modelu supervised.
- `src/training/train_active.py` - pojedynczy eksperyment Active Learning.
- `src/run_experiments.py` - uruchamianie większych siatek eksperymentów.
- `src/analysis/analyze_active_screening.py` - analiza screeningu Active Learning.
- `src/analysis/plotting.py` - funkcje do generowania wykresów.
- `src/utils/load_data.py` - ładowanie i przygotowanie danych.
- `src/utils/labels_mapping.py` - mapowanie etykiet do zadań binarnych.
- `src/utils/shared.py` - model ResNet18, predykcje, CSV, metryki i wspólne helpery.

## Wyniki Finalne

W repozytorium znajdują się trzy główne pliki wynikowe:

```text
results/final_supervised.csv
results/final_active_screening.csv
results/final_active_best.csv
```

Ich rola:

- `final_supervised.csv` - baseline supervised dla finalnych datasetów;
- `final_active_screening.csv` - screening parametrów Active Learning na `bloodmnist`;
- `final_active_best.csv` - finalne porównanie strategii Active Learning na wybranym zestawie parametrów.

## Uwagi Reprodukcyjne

Eksperymenty używają ustalonych seedów, a skrypty ustawiają ziarna losowości dla Pythona, NumPy i PyTorch. Mimo tego pełna bitowa reprodukowalność może zależeć od wersji bibliotek, urządzenia, sterowników CUDA i konfiguracji środowiska.

Najważniejsze parametry eksperymentów są zapisane w plikach YAML oraz w wynikowych CSV.
