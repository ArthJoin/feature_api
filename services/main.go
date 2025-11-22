package main

import (
	"context"
	"net/http"
	"time"
	"log"

	"feature_service/internal/repository"
	"feature_service/internal/http"

	"github.com/gin-gonic/gin"
)

func App(pg *postgres.Postgres) *gin.Engine {
	r := gin.Default()

	r.POST("/features", func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
		defer cancel()

		resultChan := make(chan error, 1)

		go func() {
			// Передаем pg в хендлер через контекст
			c.Set("pg", pg)
			handlers.Handler(c)
			resultChan <- nil // Можно доработать возврат ошибки
		}()

		select {
		case <-resultChan:
			// Ответ уже отправлен из хендлера
		case <-ctx.Done():
			c.JSON(http.StatusGatewayTimeout, gin.H{"error": "processing timeout"})
		}
	})

	return r
}

func main() {
	pg, err := postgres.NewPostgres()
	if err != nil {
		log.Fatalf("failed to init postgres: %v", err)
	}
	defer pg.Close()

	r := App(pg)

	if err := r.Run(":9000"); err != nil {
		log.Fatalf("failed to run http server: %v", err)
	}
}