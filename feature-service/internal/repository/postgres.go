package postgres

import (
	"context"
	"fmt"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"
)

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

func (p *Postgres) GetFeatures(ctx context.Context) (map[string]any, error) {
	cstDimID, ok := ctx.Value("cst_dim_id").(int64)
	if !ok {
		return nil, fmt.Errorf("cst_dim_id not found in context or has wrong type")
	}

	transDate, ok := ctx.Value("transdate").(string)
	if !ok {
		return nil, fmt.Errorf("transdate not found in context or has wrong type")
	}

	rows, err := p.pool.Query(ctx, `
		SELECT *
		FROM fs.v_txn_ml_features
		WHERE cst_dim_id = $1 AND transdate = $2
		LIMIT 1
	`, cstDimID, transDate)
	if err != nil {
		return nil, fmt.Errorf("query failed: %w", err)
	}
	defer rows.Close()

	if !rows.Next() {
		return nil, fmt.Errorf("no features found for cst_dim_id=%d, transdate=%s", cstDimID, transDate)
	}

	values, err := rows.Values()
	if err != nil {
		return nil, fmt.Errorf("rows.Values error: %w", err)
	}

	fields := rows.FieldDescriptions()
	if len(fields) != len(values) {
		return nil, fmt.Errorf("fields and values length mismatch: %d vs %d", len(fields), len(values))
	}

	result := make(map[string]any, len(values))
	for i, fd := range fields {
		colName := string(fd.Name)
		result[colName] = values[i]
	}

	return result, nil
}

func (p *Postgres) Close() {
	if p.pool != nil {
		p.pool.Close()
	}
}
