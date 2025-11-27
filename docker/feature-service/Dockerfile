# syntax=docker/dockerfile:1

FROM golang:1.25-alpine AS builder

WORKDIR /app/feature-service

COPY feature-service/go.mod feature-service/go.sum ./
RUN go mod download

COPY feature-service/ .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o server ./main.go

FROM alpine:3.20

WORKDIR /app
COPY --from=builder /app/feature-service/server .

ENV PORT=8080
EXPOSE 8080

CMD ["./server"]
