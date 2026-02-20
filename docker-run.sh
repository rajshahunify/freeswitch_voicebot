#!/bin/bash
# =============================================================================
# FreeSWITCH VoiceBot — Docker Build & Run Script
# =============================================================================
# Usage:
#   ./docker-run.sh build    — Build the Docker image
#   ./docker-run.sh run      — Run the container
#   ./docker-run.sh stop     — Stop the container
#   ./docker-run.sh logs     — View container logs
#   ./docker-run.sh status   — Check all processes
#   ./docker-run.sh test     — Run quick health checks
# =============================================================================

set -e

CONTAINER_NAME="voicebot"
IMAGE_NAME="voicebot-freeswitch"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

case "$1" in

  build)
    echo -e "${GREEN}Building Docker image...${NC}"
    docker build -t ${IMAGE_NAME}:latest -f "${SCRIPT_DIR}/Dockerfile.voicebot" "${SCRIPT_DIR}"
    echo -e "${GREEN}✓ Image built: ${IMAGE_NAME}:latest${NC}"
    ;;

  run)
    echo -e "${GREEN}Starting VoiceBot container...${NC}"

    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      echo -e "${YELLOW}Container '${CONTAINER_NAME}' already exists. Removing...${NC}"
      docker rm -f ${CONTAINER_NAME}
    fi

    docker run -d \
      --name ${CONTAINER_NAME} \
      --network host \
      --restart unless-stopped \
      -v "${SCRIPT_DIR}:/opt/voicebot" \
      -v "${SCRIPT_DIR}/sounds:/usr/local/freeswitch/sounds/custom" \
      ${IMAGE_NAME}:latest

    echo -e "${GREEN}✓ Container started: ${CONTAINER_NAME}${NC}"
    echo -e "${YELLOW}View logs: ./docker-run.sh logs${NC}"
    ;;

  stop)
    echo -e "${YELLOW}Stopping VoiceBot container...${NC}"
    docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}
    echo -e "${GREEN}✓ Container stopped and removed${NC}"
    ;;

  logs)
    docker logs -f ${CONTAINER_NAME}
    ;;

  status)
    echo -e "${GREEN}=== Container Status ===${NC}"
    docker ps --filter name=${CONTAINER_NAME} --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo -e "${GREEN}=== Supervisord Processes ===${NC}"
    docker exec ${CONTAINER_NAME} supervisorctl status
    echo ""
    echo -e "${GREEN}=== FreeSWITCH Status ===${NC}"
    docker exec ${CONTAINER_NAME} fs_cli -x "status"
    echo ""
    echo -e "${GREEN}=== mod_audio_fork ===${NC}"
    docker exec ${CONTAINER_NAME} fs_cli -x "module_exists mod_audio_fork"
    ;;

  test)
    echo -e "${GREEN}=== Running Health Checks ===${NC}"

    # Check container is running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      echo -e "${GREEN}✓ Container is running${NC}"
    else
      echo -e "${RED}✗ Container is NOT running${NC}"
      exit 1
    fi

    # Check FreeSWITCH
    FS_STATUS=$(docker exec ${CONTAINER_NAME} fs_cli -x "status" 2>/dev/null)
    if echo "$FS_STATUS" | grep -q "UP"; then
      echo -e "${GREEN}✓ FreeSWITCH is UP${NC}"
    else
      echo -e "${RED}✗ FreeSWITCH is not ready${NC}"
    fi

    # Check mod_audio_fork
    MOD_CHECK=$(docker exec ${CONTAINER_NAME} fs_cli -x "module_exists mod_audio_fork" 2>/dev/null)
    if echo "$MOD_CHECK" | grep -q "true"; then
      echo -e "${GREEN}✓ mod_audio_fork is loaded${NC}"
    else
      echo -e "${RED}✗ mod_audio_fork NOT loaded${NC}"
    fi

    # Check VoiceBot health endpoint
    HEALTH=$(curl -s http://127.0.0.1:8000/health 2>/dev/null)
    if echo "$HEALTH" | grep -q "healthy"; then
      echo -e "${GREEN}✓ VoiceBot server is healthy${NC}"
    else
      echo -e "${RED}✗ VoiceBot server not responding${NC}"
    fi

    # Check ESL connectivity
    ESL_TEST=$(docker exec ${CONTAINER_NAME} python3 -c "
from esl_manager import get_esl_manager
esl = get_esl_manager()
print(esl.send('status'))
" 2>/dev/null)
    if echo "$ESL_TEST" | grep -q "UP"; then
      echo -e "${GREEN}✓ ESL Manager connection working${NC}"
    else
      echo -e "${RED}✗ ESL Manager connection failed${NC}"
    fi

    echo -e "\n${GREEN}=== All checks complete ===${NC}"
    ;;

  *)
    echo "Usage: $0 {build|run|stop|logs|status|test}"
    echo ""
    echo "  build   — Build the Docker image"
    echo "  run     — Start the container (with host networking)"
    echo "  stop    — Stop and remove the container"
    echo "  logs    — Stream container logs"
    echo "  status  — Check all processes and FreeSWITCH status"
    echo "  test    — Run health checks (FS, mod_audio_fork, VoiceBot, ESL)"
    exit 1
    ;;
esac
