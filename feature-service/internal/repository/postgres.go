package postgres

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type Features struct {
	TransDate                   *time.Time `json:"transdate"`
	CstDimID                    *int64     `json:"cst_dim_id"`
	MonthlyOSChanges            *int       `json:"monthly_os_changes"`
	MonthlyPhoneModelChanges    *int       `json:"monthly_phone_model_changes"`
	LastPhoneModelCategorical   *string    `json:"last_phone_model_categorical"`
	LastOSCategorical           *string    `json:"last_os_categorical"`
	LoginsLast7Days             *int       `json:"logins_last_7_days"`
	LoginsLast30Days            *int       `json:"logins_last_30_days"`
	LoginFrequency7d            *float64   `json:"login_frequency_7d"`
	LoginFrequency30d           *float64   `json:"login_frequency_30d"`
	FreqChange7dVsMean          *float64   `json:"freq_change_7d_vs_mean"`
	Logins7dOver30dRatio        *float64   `json:"logins_7d_over_30d_ratio"`
	AvgLoginInterval30d         *float64   `json:"avg_login_interval_30d"`
	StdLoginInterval30d         *float64   `json:"std_login_interval_30d"`
	VarLoginInterval30d         *float64   `json:"var_login_interval_30d"`
	EwmLoginInterval7d          *float64   `json:"ewm_login_interval_7d"`
	BurstinessLoginInterval     *float64   `json:"burstiness_login_interval"`
	FanoFactorLoginInterval     *float64   `json:"fano_factor_login_interval"`
	ZScoreAvgLoginInterval7d    *float64   `json:"zscore_avg_login_interval_7d"`
}

type Postgres struct {
	pool *pgxpool.Pool
}

 
func NewPostgres() (*Postgres, error) {
	url := os.Getenv("POSTGRES_DSN")
	if url == "" {
		return nil, fmt.Errorf("POSTGRES_DSN env is empty")
	}

	pool, err := pgxpool.New(context.Background(), url)
	if err != nil {
		return nil, fmt.Errorf("failed to init pg pool: %w", err)
	}

	return &Postgres{pool: pool}, nil
}

func (p *Postgres) GetFeatures(ctx context.Context) ([]Features, error) {
	// Получаем cstDimID из контекста
	cstDimID, ok := ctx.Value("cst_dim_id").(int64)
	if !ok {
		return nil, fmt.Errorf("cst_dim_id not found in context")
	}
	transDate, ok := ctx.Value("transdate").(string)
	if !ok {
		return nil, fmt.Errorf("transdate not found in context")
	}
	rows, err := p.pool.Query(ctx, `
		SELECT *
		FROM fs.client_behavior_patterns
		WHERE cst_dim_id = $1 AND transdate = $2
	`, cstDimID, transDate)
	if err != nil {
		return nil, fmt.Errorf("query failed: %w", err)
	}
	defer rows.Close()

	var out []Features

	for rows.Next() {
		var f Features
		       err := rows.Scan(
			       &f.TransDate,
			       &f.CstDimID,
			       &f.MonthlyOSChanges,
			       &f.MonthlyPhoneModelChanges,
			       &f.LastPhoneModelCategorical,
			       &f.LastOSCategorical,
			       &f.LoginsLast7Days,
			       &f.LoginsLast30Days,
			       &f.LoginFrequency7d,
			       &f.LoginFrequency30d,
			       &f.FreqChange7dVsMean,
			       &f.Logins7dOver30dRatio,
			       &f.AvgLoginInterval30d,
			       &f.StdLoginInterval30d,
			       &f.VarLoginInterval30d,
			       &f.EwmLoginInterval7d,
			       &f.BurstinessLoginInterval,
			       &f.FanoFactorLoginInterval,
			       &f.ZScoreAvgLoginInterval7d,
		       )
		if err != nil {
			return nil, fmt.Errorf("scan error: %w", err)
		}
		out = append(out, f)
	}

	return out, nil
}

func (p *Postgres) Close() {
	if p.pool != nil {
		p.pool.Close()
	}
}


