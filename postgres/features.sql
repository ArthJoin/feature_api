DROP VIEW IF EXISTS fs.v_txn_ml_features CASCADE;

CREATE OR REPLACE VIEW fs.v_txn_ml_features AS
WITH
base AS (
    SELECT
        t.cst_dim_id,
        t.transdate,
        t.transdatetime,
        t.amount::double precision AS amount,
        t.docno,

        COALESCE(cbp.monthly_os_changes, 0)                  AS monthly_os_changes,
        COALESCE(cbp.monthly_phone_model_changes, 0)         AS monthly_phone_model_changes,
        cbp.last_phone_model_categorical,
        cbp.last_os_categorical,
        COALESCE(cbp.logins_last_7_days, 0)::double precision            AS logins_last_7_days,
        COALESCE(cbp.logins_last_30_days, 0)::double precision           AS logins_last_30_days,
        COALESCE(cbp.login_frequency_7d, 0.0)                 AS login_frequency_7d,
        COALESCE(cbp.login_frequency_30d, 0.0)                AS login_frequency_30d,
        COALESCE(cbp.freq_change_7d_vs_mean, 0.0)             AS freq_change_7d_vs_mean,
        COALESCE(cbp.logins_7d_over_30d_ratio, 0.0)           AS logins_7d_over_30d_ratio,
        COALESCE(cbp.avg_login_interval_30d, 0.0)             AS avg_login_interval_30d,
        COALESCE(cbp.std_login_interval_30d, 0.0)             AS std_login_interval_30d,
        COALESCE(cbp.var_login_interval_30d, 0.0)             AS var_login_interval_30d,
        COALESCE(cbp.ewm_login_interval_7d, 0.0)              AS ewm_login_interval_7d,
        COALESCE(cbp.burstiness_login_interval, 0.0)          AS burstiness_login_interval,
        COALESCE(cbp.fano_factor_login_interval, 0.0)         AS fano_factor_login_interval,
        COALESCE(cbp.zscore_avg_login_interval_7d, 0.0)       AS zscore_avg_login_interval_7d
    FROM fs.transactions t
    LEFT JOIN LATERAL (
        SELECT *
        FROM fs.client_behavior_patterns cbp
        WHERE cbp.cst_dim_id = t.cst_dim_id
          AND cbp.transdate <= t.transdatetime
        ORDER BY cbp.transdate DESC
        LIMIT 1
    ) cbp ON TRUE
),

cust_agg AS (
    SELECT
        cst_dim_id,
        AVG(amount::double precision)                      AS cust_amount_mean,
        COALESCE(stddev_pop(amount::double precision), 0.0) AS cust_amount_std,
        MAX(amount::double precision)                      AS cust_amount_max,
        COUNT(*)::double precision                         AS cust_txn_count
    FROM fs.transactions
    GROUP BY cst_dim_id
),

amount_stats AS (
    SELECT
        percentile_disc(0.99) WITHIN GROUP (ORDER BY amount::double precision) AS amount_q99,
        AVG(amount::double precision)                                          AS amount_mean,
        COALESCE(stddev_pop(amount::double precision), 1.0)                    AS amount_std,
        percentile_disc(0.90) WITHIN GROUP (ORDER BY amount::double precision) AS amount_high_thresh
    FROM fs.transactions
),

burst_stats AS (
    SELECT
        percentile_disc(0.90) WITHIN GROUP (ORDER BY ABS(burstiness_login_interval)) AS burst_q90
    FROM fs.client_behavior_patterns
),

cust_txn_stats AS (
    SELECT
        percentile_disc(0.75) WITHIN GROUP (ORDER BY cust_txn_count) AS cust_txn_q75,
        percentile_disc(0.90) WITHIN GROUP (ORDER BY cust_txn_count) AS cust_txn_q90
    FROM (
        SELECT cst_dim_id, COUNT(*)::double precision AS cust_txn_count
        FROM fs.transactions
        GROUP BY cst_dim_id
    ) s
),

global_stats AS (
    SELECT
        a.amount_q99,
        a.amount_mean,
        a.amount_std,
        a.amount_high_thresh,
        b.burst_q90,
        c.cust_txn_q75,
        c.cust_txn_q90
    FROM amount_stats a
    CROSS JOIN burst_stats b
    CROSS JOIN cust_txn_stats c
)

SELECT
	b.cst_dim_id,
    b.transdate,
    b.amount,
    b.docno,
    b.monthly_os_changes,
    b.monthly_phone_model_changes,
    b.last_phone_model_categorical,
    b.last_os_categorical,
    b.logins_last_7_days,
    b.logins_last_30_days,
    b.login_frequency_7d,
    b.login_frequency_30d,
    b.freq_change_7d_vs_mean,
    b.logins_7d_over_30d_ratio,
    b.avg_login_interval_30d,
    b.std_login_interval_30d,
    b.var_login_interval_30d,
    b.ewm_login_interval_7d,
    b.burstiness_login_interval,
    b.fano_factor_login_interval,
    b.zscore_avg_login_interval_7d,

    EXTRACT(HOUR FROM b.transdatetime)::int AS hour,
    EXTRACT(DOW  FROM b.transdatetime)::int AS weekday,
    CASE WHEN EXTRACT(HOUR FROM b.transdatetime) BETWEEN 0 AND 6  THEN 1 ELSE 0 END AS is_night,
    CASE WHEN EXTRACT(HOUR FROM b.transdatetime) BETWEEN 18 AND 23 THEN 1 ELSE 0 END AS is_evening,
    CASE WHEN EXTRACT(HOUR FROM b.transdatetime) BETWEEN 6 AND 11  THEN 1 ELSE 0 END AS is_morning,
    CASE WHEN EXTRACT(DOW  FROM b.transdatetime) >= 5            THEN 1 ELSE 0 END AS is_weekend,

    clip.amt_clipped                                           AS amount_clipped,
    LN(1 + clip.amt_clipped)                                  AS amount_log,

    'ALL'::text                                               AS amount_bin,

    CASE WHEN clip.amt_clipped >= g.amount_high_thresh THEN 1 ELSE 0 END AS is_high_amount,
    (clip.amt_clipped - g.amount_mean) / NULLIF(g.amount_std, 0.0)       AS amount_z_overall,

    b.logins_last_7_days  / 7.0                                AS logins_7d_per_day,
    b.logins_last_30_days / 30.0                               AS logins_30d_per_day,
    clip.amt_clipped / (1.0 + b.logins_last_30_days)           AS amount_per_login_30d,
    ABS(b.freq_change_7d_vs_mean)                              AS freq_change_7d_vs_mean_abs,
    b.login_frequency_7d / NULLIF(b.login_frequency_30d, 0.0)  AS login_freq_ratio_7_30,
    b.logins_last_7_days / NULLIF(b.logins_last_30_days, 0.0)  AS logins_ratio_7_30,
    b.std_login_interval_30d / NULLIF(b.avg_login_interval_30d, 0.0) AS interval_cv_30d,
    b.var_login_interval_30d / NULLIF(b.avg_login_interval_30d, 0.0) AS interval_fano_30d,
    b.ewm_login_interval_7d / NULLIF(b.avg_login_interval_30d, 0.0)   AS ewm_vs_avg_interval,
    ABS(b.burstiness_login_interval)                           AS burstiness_abs,
    ABS(b.zscore_avg_login_interval_7d)                        AS zscore_login_abs,
    LN(1 + clip.amt_clipped) * ABS(b.freq_change_7d_vs_mean)   AS amountlog_x_freqchange_abs,

    ca.cust_amount_mean,
    ca.cust_amount_std,
    ca.cust_amount_max,
    ca.cust_txn_count,
    clip.amt_clipped / NULLIF(ca.cust_amount_mean, 0.0)        AS amount_vs_cust_mean,
    clip.amt_clipped / NULLIF(ca.cust_amount_max, 0.0)         AS amount_vs_cust_max,
    CASE WHEN ca.cust_txn_count >= g.cust_txn_q75 THEN 1 ELSE 0 END AS cust_is_heavy_q75,
    CASE WHEN ca.cust_txn_count >= g.cust_txn_q90 THEN 1 ELSE 0 END AS cust_is_heavy_q90,

    CASE
        WHEN ABS(b.burstiness_login_interval) >= g.burst_q90
        THEN 1 ELSE 0
    END                                                        AS is_high_burst,

    CASE
        WHEN EXTRACT(HOUR FROM b.transdatetime) BETWEEN 0 AND 6
             AND clip.amt_clipped >= g.amount_high_thresh
        THEN 1 ELSE 0
    END                                                        AS night_and_high_amount,

    CASE
        WHEN EXTRACT(DOW FROM b.transdatetime) >= 5
             AND clip.amt_clipped >= g.amount_high_thresh
        THEN 1 ELSE 0
    END                                                        AS weekend_and_high_amount,

    CASE
        WHEN EXTRACT(HOUR FROM b.transdatetime) BETWEEN 0 AND 6
             AND ABS(b.burstiness_login_interval) >= g.burst_q90
        THEN 1 ELSE 0
    END                                                        AS night_and_high_burst,

    CASE
        WHEN EXTRACT(DOW FROM b.transdatetime) >= 5
             AND ABS(b.burstiness_login_interval) >= g.burst_q90
        THEN 1 ELSE 0
    END                                                        AS weekend_and_high_burst,

    -- === EXT-взаимодействия ===
    clip.amt_clipped * ABS(b.burstiness_login_interval)        AS amount_x_burst,
    clip.amt_clipped * b.fano_factor_login_interval            AS amount_x_fano,
    LN(1 + clip.amt_clipped) * b.login_frequency_7d            AS amount_x_login_freq7,
    LN(1 + clip.amt_clipped) * b.login_frequency_30d           AS amount_x_login_freq30,
    clip.amt_clipped * b.logins_last_7_days                    AS amount_x_logins_7d,
    clip.amt_clipped * b.logins_last_30_days                   AS amount_x_logins_30d,
    ABS(b.burstiness_login_interval) * b.logins_last_7_days    AS burst_x_logins_7d,
    ABS(b.burstiness_login_interval) * b.logins_last_30_days   AS burst_x_logins_30d,

    CASE
        WHEN clip.amt_clipped >= g.amount_high_thresh
             AND COALESCE(b.logins_last_30_days, 0) = 0
        THEN 1 ELSE 0
    END                                                        AS high_amount_and_no_logins30,

    CASE
        WHEN EXTRACT(HOUR FROM b.transdatetime) BETWEEN 0 AND 6
        THEN LN(1 + clip.amt_clipped) ELSE 0
    END                                                        AS night_x_amount_log,

    CASE
        WHEN EXTRACT(DOW FROM b.transdatetime) >= 5
        THEN LN(1 + clip.amt_clipped) ELSE 0
    END                                                        AS weekend_x_amount_log,

    CASE
        WHEN EXTRACT(HOUR FROM b.transdatetime) BETWEEN 18 AND 23
        THEN LN(1 + clip.amt_clipped) ELSE 0
    END                                                        AS evening_x_amount_log,

    CASE
        WHEN ca.cust_txn_count >= g.cust_txn_q75
        THEN LN(1 + clip.amt_clipped) ELSE 0
    END                                                        AS cust_heavy_q75_x_amount,

    CASE
        WHEN ca.cust_txn_count >= g.cust_txn_q75
             AND EXTRACT(HOUR FROM b.transdatetime) BETWEEN 0 AND 6
        THEN 1 ELSE 0
    END                                                        AS cust_heavy_q75_x_night,

    CASE
        WHEN ca.cust_txn_count >= g.cust_txn_q75
             AND ABS(b.burstiness_login_interval) >= g.burst_q90
        THEN 1 ELSE 0
    END                                                        AS cust_heavy_q75_x_burst,

    CASE
        WHEN clip.amt_clipped / NULLIF(ca.cust_amount_mean, 0.0) > 3.0
        THEN 1 ELSE 0
    END                                                        AS atypical_amount_vs_mean,

    CASE
        WHEN clip.amt_clipped / NULLIF(ca.cust_amount_max, 0.0) > 1.5
        THEN 1 ELSE 0
    END                                                        AS atypical_amount_vs_max,

    CASE
        WHEN (clip.amt_clipped - g.amount_mean) / NULLIF(g.amount_std, 0.0) > 2.0
        THEN 1 ELSE 0
    END                                                        AS high_amount_z_flag,

    CASE
        WHEN ABS(b.burstiness_login_interval) >= g.burst_q90
             AND clip.amt_clipped >= g.amount_high_thresh
        THEN 1 ELSE 0
    END                                                        AS high_burst_and_high_amount,

    ABS(b.burstiness_login_interval) * b.logins_7d_over_30d_ratio AS login_spiky_ratio_max,

    0.0::double precision                                       AS iforest_score,
    0.0::double precision                                       AS lof_score

FROM base b
LEFT JOIN cust_agg ca
    ON ca.cst_dim_id = b.cst_dim_id
CROSS JOIN global_stats g
CROSS JOIN LATERAL (
    SELECT LEAST(GREATEST(COALESCE(b.amount, 0.0), 0.0), g.amount_q99) AS amt_clipped
) clip;
