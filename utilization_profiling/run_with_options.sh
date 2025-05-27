#!/bin/bash

SELECTED_APP="all"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --selected_app)
            SELECTED_APP="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--selected_app <app_name>]"
            echo ""
            echo "Options:"
            echo "  --selected_app    Specify the application name to run, default is 'all'"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Available applications:"
            echo "  DeepMD, gnn, gunrock, lammps, llama2_ft, llama3_inference,"
            echo "  m-psdns, milc, openfold, pannotia, QMCPack, resnet, sgemm, all"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use $0 --help for help information"
            exit 1
            ;;
    esac
done

# Define all application list
APPS=(
    "DeepMD"
    "gnn" 
    "gunrock"
    "lammps"
    "llama2_ft"
    "llama3_inference"
    "m-psdns"
    "milc"
    "openfold"
    "pannotia"
    "QMCPack"
    "resnet"
    "sgemm"
)

# Check if specified application exists
check_app_exists() {
    local app_name=$1
    if [ ! -d "$app_name" ]; then
        echo "Error: Application directory '$app_name' does not exist"
        return 1
    fi
    
    if [ ! -f "$app_name/run.sh" ]; then
        echo "Error: run.sh file not found in '$app_name' directory"
        return 1
    fi
    
    if [ ! -x "$app_name/run.sh" ]; then
        echo "Warning: $app_name/run.sh is not executable, adding execute permission..."
        chmod +x "$app_name/run.sh"
    fi
    
    return 0
}

# Run single application
run_single_app() {
    local app_name=$1
    echo "========================================"
    echo "Starting application: $app_name"
    echo "========================================"
    
    if check_app_exists "$app_name"; then
        cd "$app_name" || exit 1
        echo "Executing command: ./run.sh"
        ./run.sh
        local exit_code=$?
        cd ..
        
        if [ $exit_code -eq 0 ]; then
            echo "✓ $app_name completed successfully"
        else
            echo "✗ $app_name failed (exit code: $exit_code)"
        fi
        
        echo "========================================"
        echo "$app_name execution completed"
        echo "========================================"
        echo ""
        
        return $exit_code
    else
        return 1
    fi
}

# Run all applications
run_all_apps() {
    echo "========================================"
    echo "Starting all application experiments"
    echo "========================================"
    echo ""
    
    local failed_apps=()
    local successful_apps=()
    
    for app in "${APPS[@]}"; do
        if run_single_app "$app"; then
            successful_apps+=("$app")
        else
            failed_apps+=("$app")
        fi
    done
    
    echo "========================================"
    echo "All experiments completed"
    echo "========================================"
    echo "Successfully completed applications (${#successful_apps[@]}):"
    for app in "${successful_apps[@]}"; do
        echo "  ✓ $app"
    done
    
    if [ ${#failed_apps[@]} -gt 0 ]; then
        echo ""
        echo "Failed applications (${#failed_apps[@]}):"
        for app in "${failed_apps[@]}"; do
            echo "  ✗ $app"
        done
        return 1
    fi
    
    return 0
}

# Main logic
echo "Starting experiment runner script..."
echo "Selected application: $SELECTED_APP"
echo ""

if [ "$SELECTED_APP" = "all" ]; then
    run_all_apps
else
    # Check if specified application is in the list
    found=false
    for app in "${APPS[@]}"; do
        if [ "$app" = "$SELECTED_APP" ]; then
            found=true
            break
        fi
    done
    
    if [ "$found" = false ]; then
        echo "Error: Unknown application name '$SELECTED_APP'"
        echo "Available applications: ${APPS[*]}"
        exit 1
    fi
    
    run_single_app "$SELECTED_APP"
fi

exit_code=$?
echo "Script execution completed with exit code: $exit_code"
exit $exit_code