package handlers

import (
	"net/http"
	"context"
	"github.com/gin-gonic/gin"
	"feature_service/internal/repository"
)

type TransactionInput struct {
	CstDimID      int64   `json:"cst_dim_id" binding:"required"`
	TransDate     string  `json:"transdate" binding:"required"`
	TransDateTime string  `json:"transdatetime" binding:"required"`
	Amount        float64 `json:"amount" binding:"required"`
	DocNo         string  `json:"docno" binding:"required"`
	Direction     int     `json:"direction" binding:"required"`
	Target        string  `json:"target" binding:"required"`
}

func Handler(c *gin.Context) {
	var input TransactionInput

	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Получаем pg из контекста
	pgRaw, ok := c.Get("pg")
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "no pg connection"})
		return
	}
	pg, ok := pgRaw.(*postgres.Postgres)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "invalid pg type"})
		return
	}

	ctx := context.WithValue(c.Request.Context(), "cst_dim_id", input.CstDimID)
	ctx = context.WithValue(ctx, "transdate", input.TransDate)
	features, err := pg.GetFeatures(ctx)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"features": features,
	})
}
