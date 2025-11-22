-- Создаём базу feature_store
CREATE DATABASE feature_store
    WITH 
    OWNER = feature_user
    ENCODING = 'UTF8'
    TEMPLATE = template0;

-- Переключимся на БД
\connect feature_store;

-- Схема
CREATE SCHEMA IF NOT EXISTS fs AUTHORIZATION feature_user;

-- Тест табличка
CREATE TABLE IF NOT EXISTS fs.healthcheck (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP DEFAULT NOW()
);

-- Создаем таблицу для хранения фичей
CREATE TABLE fs.client_behavior_patterns (
    transdate                      timestamp without time zone,
    cst_dim_id                     bigint,
    monthly_os_changes             integer,
    monthly_phone_model_changes    integer,
    last_phone_model_categorical   text,
    last_os_categorical            text,
    logins_last_7_days             integer,
    logins_last_30_days            integer,
    login_frequency_7d             double precision,
    login_frequency_30d            double precision,
    freq_change_7d_vs_mean         double precision,
    logins_7d_over_30d_ratio       double precision,
    avg_login_interval_30d         double precision,
    std_login_interval_30d         double precision,
    var_login_interval_30d         double precision,
    ewm_login_interval_7d          double precision,
    burstiness_login_interval      double precision,
    fano_factor_login_interval     double precision,
    zscore_avg_login_interval_7d   double precision
);

-- Индекс на client_behavior_patterns
CREATE INDEX idx_cbp_cst_dim_id_transdate
    ON fs.client_behavior_patterns (cst_dim_id, transdate);

CREATE INDEX idx_client_behavior_patterns_transdate
    ON fs.client_behavior_patterns (transdate);

CREATE INDEX idx_client_behavior_patterns_cst_dim_id
    ON fs.client_behavior_patterns (cst_dim_id);


-- Таблица для хранения транзакций
CREATE TABLE fs.transactions (
    cst_dim_id        bigint,                     -- уникальный идентификатор клиента
    transdate         date,                       -- дата совершенной транзакции
    transdatetime     timestamp without time zone,-- дата и время транзакции
    amount            numeric(18,2),              -- сумма операции
    docno             bigint,                     -- уникальный идентификатор транзакции
    direction         text,                       -- зашифрованный id получателя/адресата
    target            integer                     -- 1 = мошенничество, 0 = чистая транзакция
);

CREATE INDEX idx_client_transactions_cst_dim_id
    ON fs.transactions (cst_dim_id);

CREATE INDEX idx_client_transactions_transdate
    ON fs.transactions (transdate);

CREATE INDEX idx_client_transactions_cst_date
    ON fs.transactions (cst_dim_id, transdatetime);


-- Загрузим данные
COPY fs.transactions
FROM '/var/lib/postgresql/dump/transactions.csv'
CSV HEADER;

COPY fs.client_behavior_patterns
FROM '/var/lib/postgresql/dump/features.csv'
CSV HEADER;