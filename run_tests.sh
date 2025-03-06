#!/bin/bash

# Make the script exit if any command fails
set -e

# Function to display help
display_help() {
    echo "LLM Evaluation Framework Test Runner"
    echo ""
    echo "Usage: ./run_tests.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help        Display this help message"
    echo "  -a, --all         Run all tests (default)"
    echo "  -u, --unit        Run only unit tests"
    echo "  -c, --coverage    Run tests with coverage reporting"
    echo "  -d, --docker      Run tests in a Docker container"
    echo "  -m MODULE, --module MODULE    Run tests for a specific module"
    echo ""
    echo "Examples:"
    echo "  ./run_tests.sh                 # Run all tests"
    echo "  ./run_tests.sh -c              # Run all tests with coverage"
    echo "  ./run_tests.sh -m test_core_models  # Run only core models tests"
    echo "  ./run_tests.sh -d              # Run all tests in Docker"
}

# Default options
RUN_ALL=true
RUN_COVERAGE=false
RUN_DOCKER=false
SPECIFIC_MODULE=""

# Parse command line arguments
while [ "$1" != "" ]; do
    case $1 in
        -h | --help )
            display_help
            exit 0
            ;;
        -a | --all )
            RUN_ALL=true
            ;;
        -u | --unit )
            RUN_ALL=false
            ;;
        -c | --coverage )
            RUN_COVERAGE=true
            ;;
        -d | --docker )
            RUN_DOCKER=true
            ;;
        -m | --module )
            shift
            SPECIFIC_MODULE=$1
            RUN_ALL=false
            ;;
        * )
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
    shift
done

# Run tests in Docker if specified
if [ "$RUN_DOCKER" = true ]; then
    echo "Building test Docker image..."
    docker build -f Dockerfile.test -t llm-eval-test .
    
    echo "Running tests in Docker..."
    if [ -n "$SPECIFIC_MODULE" ]; then
        docker run llm-eval-test python -m unittest tests.$SPECIFIC_MODULE
    elif [ "$RUN_COVERAGE" = true ]; then
        docker run llm-eval-test bash -c "coverage run -m unittest discover -s tests && coverage report -m"
    else
        docker run llm-eval-test
    fi
    exit 0
fi

# Run tests locally
if [ "$RUN_ALL" = true ] && [ "$RUN_COVERAGE" = true ]; then
    echo "Running all tests with coverage..."
    coverage run -m unittest discover -s tests
    coverage report -m
    echo "For detailed coverage report, run: coverage html"
elif [ "$RUN_ALL" = true ]; then
    echo "Running all tests..."
    python -m unittest discover -s tests
elif [ -n "$SPECIFIC_MODULE" ] && [ "$RUN_COVERAGE" = true ]; then
    echo "Running tests for $SPECIFIC_MODULE with coverage..."
    coverage run -m unittest tests.$SPECIFIC_MODULE
    coverage report -m
elif [ -n "$SPECIFIC_MODULE" ]; then
    echo "Running tests for $SPECIFIC_MODULE..."
    python -m unittest tests.$SPECIFIC_MODULE
fi

echo "All tests completed successfully!"
