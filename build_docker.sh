#!/bin/bash
# Build and run Docker image for LSTM + LLM Ablation Study

set -e

echo "=================================================="
echo "LSTM + LLM Ablation Study - Docker Build Script"
echo "=================================================="

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Build image
echo -e "${BLUE}[1/4] Building Docker image...${NC}"
docker build -t lstm-llm:latest . --progress=plain

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image built successfully${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Step 2: Get image size
echo -e "${BLUE}[2/4] Image information:${NC}"
docker images lstm-llm:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Step 3: Run container
echo -e "${BLUE}[3/4] Running container...${NC}"
echo "This will:"
echo "  1. Start Ollama service"
echo "  2. Download gemma:4b model (~2GB)"
echo "  3. Run ablation_pipeline.py --variants all"
echo ""
echo "Note: First run will take 10-15 minutes due to model download"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker run -it --rm \
        -v "$(pwd)/results:/app/results" \
        -p 11434:11434 \
        lstm-llm:latest
    
    echo -e "${GREEN}✓ Pipeline completed${NC}"
    
    # Step 4: Show results
    echo -e "${BLUE}[4/4] Results saved to:${NC}"
    echo "  - results/ablation_study/ABLATION_COMPARISON.md"
    echo "  - results/ablation_study/plots/"
else
    echo "Cancelled"
    exit 1
fi

echo -e "${GREEN}✓ Done!${NC}"
