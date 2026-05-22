# data/

Подготовленный (processed) датасет публикуется в составе репозитория:
`processed/floodit_final.parquet` — 222 строки × 60 колонок, 111 дней
наблюдений по двум мобильным приложениям («Flood-It!» и «Flood-It! 2»).

Сырьё (raw) — выгрузки из Firebase Public Data Program (BigQuery), Google
Play Scraper, Kaggle-снапшота категории Google Play, Google Trends и
Wikipedia Pageviews — в репозиторий не включено, поскольку:

1. суммарный объём превышает 1,6 ГБ;
2. источники открытые и воспроизводимые из публичных API.

Подготовленный датасет — единственная точка входа для всех расчётов и
не пересобирается в штатном режиме работы системы; его SHA-256 зафиксирован
в реестре `vkr/v2/artifacts/registry.csv` и проверяется командой
`python scripts/check_registry.py`.
