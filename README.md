# Krótkoterminowa prognoza temperatury – model globalny XGBoost

Projekt przedstawia system krótkoterminowej prognozy średniej dobowej temperatury powietrza
w horyzoncie od 1 do 7 dni, oparty na globalnym modelu uczenia maszynowego XGBoost.

Rozwiązanie zostało zaprojektowane z myślą o pracy na dużych zbiorach danych
oraz porównaniu jakości predykcji z klasycznymi modelami odniesienia.

---

## Dane

Wykorzystano publicznie dostępne dobowe dane obserwacyjne
Instytutu Meteorologii i Gospodarki Wodnej – Państwowego Instytutu Badawczego (IMGW-PIB),
obejmujące lata **1951–2024** oraz **342 stacje pomiarowe**.

Dane udostępniane są w formacie CSV i mogą być wykorzystywane w celach naukowych
po wskazaniu źródła, zgodnie z regulaminem IMGW-PIB.

---

## Metodyka

- budowa globalnego zbioru cech dla wszystkich stacji jednocześnie
- inżynieria cech: sezonowość, klimatologia, opóźnienia czasowe
- walidacja czasowa bez wycieku informacji
- porównanie z modelami odniesienia (persistence, klimatologia)
- porównanie z innymi podejściami ML/DL (XGBoost lokalny i sezonowy, LSTM, TFT)
- kalibracja predykcji względem komponentu klimatycznego

---

## Wyniki

Najlepsze rezultaty uzyskano dla **globalnego modelu XGBoost**:

- **MAE ≈ 2,66 °C**
- **RMSE ≈ 3,38 °C**
- średnia poprawa względem modelu persistence:
  - ΔMAE ≈ 0,97 °C
  - ΔRMSE ≈ 1,25 °C

Model globalny zapewnia stabilną jakość predykcji dla wielu lokalizacji
przy umiarkowanej złożoności obliczeniowej.

---

## Technologie

- Python
- pandas, scikit-learn
- XGBoost
- TensorFlow / LSTM
- SQL / MySQL

---

## Struktura projektu (uproszczona)

- `Models/` – implementacja modelu globalnego XGBoost  
- `Scripts/` – przygotowanie danych i ekstrakcja cech  
- `Metadata/` – konfiguracja środowiska i parametrów  

---

## Uruchomienie

1. Utworzenie środowiska:
   ```bash
   conda env create -f environment.yaml
   conda activate climate-ml

2. Budowa cech:
   ```bash
   python Scripts/build_global_features.py

3. Trening modelu:
   ```bash
   python Models/xgb_global.py
