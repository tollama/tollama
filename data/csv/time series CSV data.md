아래는 **time series CSV 데이터를 바로 받을 수 있는 URL 중심** 정리입니다. `pd.read_csv(url)`로 바로 읽히는 것, 또는 `zip/csv.gz/txt`지만 CSV처럼 읽을 수 있는 것도 포함했습니다.

## **1. TSFM / Long-horizon forecasting benchmark용**

| **데이터**                  | **도메인 / 주기**    | **URL**                                                                                     |
| ------------------------ | --------------- | ------------------------------------------------------------------------------------------- |
| ETT original `ETTh1.csv` | 전력 변압기 / hourly | [ETTh1.csv](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv) |
| ETT original `ETTh2.csv` | 전력 변압기 / hourly | [ETTh2.csv](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv) |
| ETT original `ETTm1.csv` | 전력 변압기 / 15-min | [ETTm1.csv](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv) |
| ETT original `ETTm2.csv` | 전력 변압기 / 15-min | [ETTm2.csv](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv) |

ETDataset는 long sequence forecasting용으로 수집된 전력 변압기 데이터이며, 사전처리되어 CSV로 제공되고 2016년 7월부터 2018년 7월까지의 데이터를 포함합니다.  

## **2. Autoformer / Informer 계열 표준 벤치마크 모음**

| **데이터**                | **도메인**  | **URL**                                                                                                                            |
| ---------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `electricity.csv`      | 전력 수요    | [electricity.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/electricity.csv)           |
| `traffic.csv`          | 교통 센서    | [traffic.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/traffic.csv)                   |
| `weather.csv`          | 기상       | [weather.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/weather.csv)                   |
| `exchange_rate.csv`    | 환율       | [exchange_rate.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/exchange_rate.csv)       |
| `national_illness.csv` | ILI / 질병 | [national_illness.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/national_illness.csv) |
| `ETTh1.csv`            | 전력 변압기   | [ETTh1.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh1.csv)                       |
| `ETTh2.csv`            | 전력 변압기   | [ETTh2.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh2.csv)                       |
| `ETTm1.csv`            | 전력 변압기   | [ETTm1.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTm1.csv)                       |
| `ETTm2.csv`            | 전력 변압기   | [ETTm2.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTm2.csv)                       |

이 Hugging Face 데이터셋은 ETT, Electricity, Exchange Rate, Traffic, Weather, ILI 등 Autoformer 계열에서 많이 쓰는 벤치마크 파일들을 포함합니다.  

## **3. 에너지 / 전력 수요 데이터**

| **데이터**                        | **내용**                                 | **URL**                                                                                                                                     |
| ------------------------------ | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| UCI Electricity Load Diagrams  | 370 고객 전력 사용량, 15분 단위                  | [electricityloaddiagrams20112014.zip](https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip)                    |
| UCI Individual Household Power | 한 가구 전력 소비, 1분 단위                      | [individual household power.zip](https://archive.ics.uci.edu/static/public/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip) |
| OPSD Europe hourly             | 유럽 전력 load, price, wind, solar, hourly | [time_series_60min_singleindex.csv](https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv)       |
| OPSD Europe 15-min             | 일부 국가/항목 15분 단위                        | [time_series_15min_singleindex.csv](https://data.open-power-system-data.org/time_series/2020-10-06/time_series_15min_singleindex.csv)       |
| Victoria electricity           | 호주 Victoria 전력수요, 30분 단위               | [vic_electricity.csv](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/vic_electricity.csv)                       |

UCI Electricity Load Diagrams는 15분 단위 전력 사용량 데이터이고, UCI Household Power는 약 4년간의 1분 단위 가정 전력 측정 데이터입니다.   OPSD는 유럽 32개국 전력 load, 가격, 태양광, 풍력 데이터를 hourly 중심으로 제공하며 CSV 파일도 직접 제공합니다.  

## **4. 기상 / 환경 데이터**

| **데이터**              | **내용**                       | **URL**                                                                                                                         |
| -------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Jena Climate         | 14개 기상 변수, 10분 단위, 2009–2016 | [jena_climate_2009_2016.csv.zip](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)    |
| Meteostat hourly     | 관측소별 연도별 hourly CSV gzip     | `https://data.meteostat.net/hourly/{year}/{station}.csv.gz`                                                                     |
| Meteostat daily      | 관측소별 연도별 daily CSV gzip      | `https://data.meteostat.net/daily/{year}/{station}.csv.gz`                                                                      |
| Meteostat monthly    | 관측소별 monthly CSV gzip        | `https://data.meteostat.net/monthly/{station}.csv.gz`                                                                           |
| Valencia air quality | 대기오염 hourly                  | [air_quality_valencia.csv](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/air_quality_valencia.csv) |

TensorFlow 튜토리얼의 Jena Climate 데이터는 Max Planck Institute 기상 관측소 데이터로, 14개 feature가 10분 단위로 수집되었고 다운로드 URL도 명시되어 있습니다.   Meteostat은 가입 없이 관측소별 time series CSV bulk dump를 제공합니다.  

## **5. 금융 / 경제 / 거시 시계열**

| **데이터**                     | **예시**             | **URL**                                                                                                 |
| --------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------- |
| FRED CPI                    | 미국 CPI             | [CPIAUCSL CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL)                             |
| FRED unemployment           | 미국 실업률             | [UNRATE CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE)                                 |
| Stooq AAPL daily            | 주가 OHLCV           | [AAPL daily CSV](https://stooq.com/q/d/l/?s=aapl.us&i=d)                                                |
| Stooq S&P500 daily          | 지수 OHLCV           | [S&P 500 daily CSV](https://stooq.com/q/d/l/?s=%5Espx&i=d)                                              |
| OWID electricity generation | 국가별 발전량            | [electricity-generation.csv](https://ourworldindata.org/grapher/electricity-generation.csv)             |
| OWID global GDP             | 장기 GDP             | [global-gdp-over-the-long-run.csv](https://ourworldindata.org/grapher/global-gdp-over-the-long-run.csv) |
| World Bank GDP per capita   | 국가별 GDP per capita | [World Bank CSV zip](https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.CD?downloadformat=csv)       |

FRED는 그래프 다운로드에서 CSV를 지원하고, API에서도 `file_type=csv`를 제공합니다.   Stooq는 daily, hourly, 5-minute 시장 데이터를 CSV로 제공하는 무료 데이터 소스입니다.   OWID는 chart별 CSV Data URL을 제공합니다.  

## **6. 가벼운 튜토리얼 / 실험용 CSV**

| **데이터**          | **내용**                    | **URL**                                                                                                                                                     |
| ---------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AirPassengers    | 월별 항공 승객                  | [air_passengers.csv](https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv)                               |
| h2o              | 월별 의약품 지출                 | [h2o.csv](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/h2o.csv)                                                               |
| Bike sharing     | 시간별 자전거 대여 + 날씨           | [bike_sharing_dataset_clean.csv](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/bike_sharing_dataset_clean.csv)                 |
| Website visits   | 일별 웹사이트 방문                | [visitas_por_dia_web_cienciadedatos.csv](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/visitas_por_dia_web_cienciadedatos.csv) |
| Wikipedia visits | Peyton Manning page views | [wikipedia_visits.csv](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/wikipedia_visits.csv)                                     |
| UK daily flights | 영국 일별 항공편 수               | [uk_daily_flights.csv](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/uk_daily_flights.csv)                                     |

skforecast-datasets는 튜토리얼과 예제용 time series CSV를 직접 다운로드 가능한 raw URL로 정리해 두고 있습니다.  

## **Tollama / TSFM 평가용으로 우선 추천**

가장 실용적인 조합은 다음입니다.

1. **기본 벤치마크**: ETT + Electricity + Traffic + Weather + Exchange + ILI
2. **에너지 특화**: OPSD + UCI Electricity + Victoria electricity
3. **기상 covariate 실험**: Jena Climate + Meteostat
4. **간단한 sanity check**: AirPassengers + h2o + Bike Sharing

`pd.read_csv()` 테스트 예시는 아래처럼 시작하면 됩니다.

```python
import pandas as pd

url = "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh1.csv"
df = pd.read_csv(url)

print(df.shape)
print(df.head())
```
