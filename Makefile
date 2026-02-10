.PHONY: install dev test lint format docker-up docker-down seed scan demo clean help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)LedgerGuard - Business Reliability Engine$(NC)"
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install all dependencies (backend + frontend)
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	pip install -e ".[dev]"
	@echo "$(BLUE)Installing frontend dependencies...$(NC)"
	cd frontend && npm install
	@echo "$(GREEN)Installation complete!$(NC)"

dev: ## Start development servers (backend + frontend in parallel)
	@echo "$(BLUE)Starting development environment...$(NC)"
	@echo "$(YELLOW)Backend: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:3000$(NC)"
	@echo "$(YELLOW)Docs: http://localhost:8000/docs$(NC)"
	@make -j2 dev-backend dev-frontend

dev-backend: ## Start FastAPI backend with hot reload
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend: ## Start Vite frontend dev server
	cd frontend && npm run dev

test: ## Run all tests with coverage
	@echo "$(BLUE)Running test suite...$(NC)"
	pytest tests/ --cov=api --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(NC)"

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-golden: ## Run golden path tests
	pytest tests/golden/ -v

lint: ## Run linters (ruff + mypy)
	@echo "$(BLUE)Running linters...$(NC)"
	ruff check api/ tests/
	mypy api/

format: ## Format code with black and ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	black api/ tests/ scripts/
	ruff check --fix api/ tests/ scripts/
	cd frontend && npm run format

docker-up: ## Start all Docker services
	@echo "$(BLUE)Starting Docker services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "$(YELLOW)API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:3000$(NC)"
	@echo "$(YELLOW)Redis: localhost:6379$(NC)"

docker-down: ## Stop all Docker services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	docker-compose down
	@echo "$(GREEN)Services stopped!$(NC)"

docker-logs: ## Show Docker logs
	docker-compose logs -f

docker-rebuild: ## Rebuild and restart Docker services
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d

seed: ## Seed sandbox data from QuickBooks
	@echo "$(BLUE)Seeding sandbox data...$(NC)"
	python scripts/seed_sandbox.py
	@echo "$(GREEN)Seeding complete!$(NC)"

scan: ## Run full reliability scan
	@echo "$(BLUE)Running reliability scan...$(NC)"
	python -m api.cli scan --full
	@echo "$(GREEN)Scan complete!$(NC)"

demo: ## Run complete demo flow
	@echo "$(BLUE)Running demo scenario...$(NC)"
	python scripts/demo_run.py
	@echo "$(GREEN)Demo complete!$(NC)"

migrate: ## Run database migrations
	@echo "$(BLUE)Running migrations...$(NC)"
	python -m api.storage.migrate
	@echo "$(GREEN)Migrations complete!$(NC)"

clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist build *.egg-info
	cd frontend && rm -rf node_modules dist .vite
	@echo "$(GREEN)Cleanup complete!$(NC)"

db-shell: ## Open DuckDB shell
	duckdb ./data/bre.duckdb

redis-cli: ## Open Redis CLI
	docker-compose exec redis redis-cli

api-shell: ## Open Python shell with API context
	python -c "from api.main import app; from api.config import get_settings; import IPython; IPython.embed()"

logs-api: ## Tail API logs
	docker-compose logs -f api

logs-worker: ## Tail worker logs
	docker-compose logs -f worker

build-frontend: ## Build frontend for production
	cd frontend && npm run build

serve-frontend: ## Serve production frontend build
	cd frontend && npm run preview

docs: ## Generate API documentation
	@echo "$(BLUE)API docs available at:$(NC)"
	@echo "$(YELLOW)http://localhost:8000/docs (Swagger UI)$(NC)"
	@echo "$(YELLOW)http://localhost:8000/redoc (ReDoc)$(NC)"

check: lint test ## Run all checks (lint + test)

ci: clean install check ## Run CI pipeline locally

.DEFAULT_GOAL := help
