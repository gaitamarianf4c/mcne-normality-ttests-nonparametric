# Teste statistice – Curs 2, temă

Gaita Marian Gabriel

Problemele 1–5 din tema de la Curs 2. Rezolvările de mână sunt în `rezolvari-de-mana/`, iar în `notebooks/tema_curs2.ipynb` am verificat rezultatele cu `scipy.stats`.

## Rezultate

| Problemă | Test | Rezultat | Decizie (α = 0.05) |
|---|---|---|---|
| 1 | Lilliefors | D = 0.222 < 0.242 | nu respingem normalitatea |
| 1 | Shapiro–Wilk | W = 0.838 < 0.859 | respingem normalitatea |
| 2 | t-test un eșantion | t(11) = 4.25, p = 0.001 | respingem H₀ |
| 2 | testul semnelor | 10+ / 2−, p = 0.039 | respingem H₀ |
| 2 | Wilcoxon signed-rank | W = 3.5 ≤ 13 | respingem H₀ |
| 3 | t-test pereche | t(9) = 5.42, p < 0.001 | respingem H₀ |
| 3 | t-test independent | t(18) = 1.34 < 2.101 | nu respingem H₀ |
| 4 | t-test independent | t(20) = −3.71, p = 0.001, IC 95%: 1.34 – 4.79 zile | respingem H₀ |
| 4 | Mann–Whitney | U = 15 ≤ 29 | respingem H₀ |

## Fișiere

- `rezolvari-de-mana/` – rezolvările de mână, câte un PDF pentru fiecare problemă
- `notebooks/tema_curs2.ipynb` – verificarea cu `scipy.stats`
- `src/curs2_stats/` – calculele de pe foaie, scrise în Python
- `tests/` – teste care compară calculele cu `scipy.stats`

## Rulare

```bash
pip install -r requirements.txt
pip install -e .
jupyter notebook notebooks/tema_curs2.ipynb
pytest
```

## Observații

- Valorile critice pentru Wilcoxon (13), Mann–Whitney (29) și t cu df = 18 (2.101) nu sunt date în enunț, le-am luat din tabele.
- W calculat cu coeficienții din tabel diferă puțin de `scipy.stats.shapiro`, pentru că scipy aproximează coeficienții.
- Pentru Lilliefors am folosit `scipy.stats.goodness_of_fit`; în curs apare `lilliefors` din statsmodels.
