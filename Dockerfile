# Build stage — pinned Go 1.26 to match go.mod requirements. Multistage to
# keep the final image distroless-small.
FROM golang:1.26-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /app/orchestrator ./cmd/orchestrator

# Final stage — Alpine for shell debugging (not distroless yet; can swap later).
FROM alpine

WORKDIR /app

COPY --from=builder /app/orchestrator /app/orchestrator

EXPOSE 8080

ENTRYPOINT ["/app/orchestrator"]
CMD ["--http", ":8080"]
