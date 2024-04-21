#!/bin/bash

TMP_DIR="/tmp/faceswap_install"

URL_CONDA="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-"
DL_CONDA="${URL_CONDA}x86_64.sh"
DL_FACESWAP="https://github.com/deepfakes/faceswap.git"
DL_XQUARTZ="https://github.com/XQuartz/XQuartz/releases/latest/download/XQuartz-2.8.5.pkg"

CONDA_PATHS=("/opt" "$HOME")
CONDA_NAMES=("anaconda" "miniconda" "miniforge")
CONDA_VERSIONS=("3" "2")
CONDA_BINS=("/bin/conda" "/condabin/conda")
DIR_CONDA="$HOME/miniconda3"
CONDA_EXECUTABLE="${DIR_CONDA}/bin/conda"
CONDA_TO_PATH=false
ENV_NAME="faceswap"
PYENV_VERSION="3.10"

DIR_FACESWAP="$HOME/faceswap"
VERSION="nvidia"

DESKTOP=false
XQUARTZ=false

header() {
    # Format header text
    length=${#1}
    padding=$(( (72 - length) / 2))
    sep=$(printf '=%.0s' $(seq 1 $padding))
    echo ""
    echo $'\e[32m'$sep $1 $sep
}

info () {
    # output info message
    while read -r line ; do
        echo $'\e[32mINFO\e[39m    '$line
    done <<< "$(echo "$1" | fmt -s -w 70)"
}

warn () {
    # output warning message
    while read -r line ; do
        echo $'\e[33mWARNING\e[39m '$line
    done <<< "$(echo "$1" | fmt -s -w 70)"
}

error () {
    # output error message.
    while read -r line ; do
        echo $'\e[31mERROR\e[39m   '$line
    done <<< "$(echo "$1" | fmt -s -w 70)"
}

yellow () {
    # Change text color to yellow
    echo $'\e[33m'
}

check_file_exists () {
    # Check whether a file exists and return true or false
    test -f "$1"
}

check_folder_exists () {
    # Check whether a folder exists and return true or false
    test -d "$1"
}

download_file () {
    # Download a file to the temp folder
    fname=$(basename -- "$1")
    curl -L "$1" --output "$TMP_DIR/$fname" --progress-bar
}

check_for_sudo() {
    # Ensure user isn't running as sudo/root. We don't want to screw up any system install
    if [ "$EUID" == 0 ] ; then
        error "This install script should not be run with root privileges. Please run as a normal user."
        exit 1
    fi
}

check_for_curl() {
    # Ensure that curl is available on the system
    if ! command -V curl &> /dev/null ; then
        error "'curl' is required for running the Faceswap installer, but could not be found. \
        Please install 'curl' before proceeding."
        exit 1
    fi
}

check_for_xcode() {
    # Ensure that xcode command line tools are available on the system
    if xcode-select -p 2>&1 | grep -q "xcode-select: error" ; then
        error "Xcode is required to install faceswap. Please install Xcode Command Line Tools \
        before proceeding. If the Xcode installer does not automatically open, then \
        you can run the command:"
        error "xcode-select --install"
        echo ""
        xcode-select --install
        exit 1
    fi
}

create_tmp_dir() {
    TMP_DIR="$(mktemp -d)"
    if [ -z "$TMP_DIR" -o ! -d "$TMP_DIR" ]; then
        # This shouldn't happen, but just in case to prevent the tmp cleanup function to mess things up.
        error "Failed creating the temporary install directory."
        exit 2
    fi
    trap cleanup_tmp_dir EXIT
}

cleanup_tmp_dir() {
  rm -rf "$TMP_DIR"
}

ask () {
    # Ask for input. First parameter: Display text, 2nd parameter variable name
    default="${!2}"
    read -rp $'\e[35m'"$1 [default: '$default']: "$'\e[39m' inp
    inp="${inp:-${default}}"
    if [ "$inp" == "\n" ] ; then inp=${!2} ; fi
    printf -v $2 "$inp"
}

ask_yesno () {
    # Ask yes or no. First Param: Question, 2nd param: Default
    # Returns True for yes, False for No
    case $2 in
        [Yy]* ) opts="[YES/no]" ;;
        [Nn]* ) opts="[yes/NO]" ;;
    esac
    while true; do
        read -rp $'\e[35m'"$1 $opts: "$'\e[39m' yn
        yn="${yn:-${2}}"
        case $yn in
            [Yy]* ) retval=true ; break ;;
            [Nn]* ) retval=false ; break ;;
            * ) echo "Please answer yes or no." ;;
        esac
    done
    $retval
}


ask_version() {
    # Ask which version of faceswap to install
    while true; do
        default=1
        read -rp $'\e[35mSelect:\t1: Apple Silicon\n\t2: NVIDIA\n\t3: CPU\n'"[default: $default]: "$'\e[39m' vers
        vers="${vers:-${default}}"
        case $vers in
            1) VERSION="apple_silicon" ; break ;;
            2) VERSION="nvidia" ; break ;;
            3) VERSION="cpu" ; break ;;
            * ) echo "Invalid selection." ;;
        esac
    done
}

banner () {
    echo $'      \e[32m                   001'
    echo $'      \e[32m                  11 10  010'
    echo $'      \e[39m         @@@@\e[32m      10'
    echo $'      \e[39m      @@@@@@@@\e[32m         00     1'
    echo $'      \e[39m    @@@@@@@@@@\e[32m  1  1            0'
    echo $'      \e[39m  @@@@@@@@\e[32m    0000 01111'
    echo $'      \e[39m @@@@@@@@@@\e[32m    01  110 01  1'
    echo $'      \e[39m@@@@@@@@@@@@\e[32m 111    010    0'
    echo $'      \e[39m@@@@@@@@@@@@@@@@\e[32m  10    0'
    echo $'      \e[39m@@@@@@@@@@@@@\e[32m   0010   1'
    echo $'      \e[39m@@@@@@@@@  @@@\e[32m   100         1'
    echo $'      \e[39m@@@@@@@ .@@@@\e[32m  10       1'
    echo $'      \e[39m #@@@@@@@@@@@\e[32m  001       0'
    echo $'      \e[39m   @@@@@@@@@@@  ,'
    echo '         @@@@@@@@  @@@@@'
    echo '        @@@@@@@@ @@@@@@@@    _'
    echo '       @@@@@@@@@,@@@@@@@@  / _|'
    echo '       %@@@@@@@@@@@@@@@@@ | |_  ___ '
    echo '           @@@@@@@@@@@@@@ |  _|/ __|'
    echo '            @@@@@@@@@@@@  | |  \__ \'
    echo '             @@@@@@@@@@(  |_|  |___/'
    echo '                @@@@@@'
    echo '                 @@@@'
    sleep 2
}

find_conda_install() {
    if check_conda_path;
        then true
        elif check_conda_locations ; then true
        else false
    fi
}

set_conda_dir_from_bin() {
    # Set the DIR_CONDA variable from the bin file
    pth="$(dirname "$1")/.."
    DIR_CONDA=$(python -c "import os, sys; print(os.path.realpath('$pth'))")
    info "Found existing conda install at: $DIR_CONDA"
}

check_conda_path()  {
    # Check if conda is in PATH
    conda_bin="$(which conda 2>/dev/null)"
    if [[ "$?" == "0" ]]; then
        set_conda_dir_from_bin "$conda_bin"
        CONDA_EXECUTABLE="$conda_bin"
        true
    else
        false
    fi
}

check_conda_locations() {
    # Check common conda install locations
    retval=false
    for path in "${CONDA_PATHS[@]}"; do
        for name in "${CONDA_NAMES[@]}" ; do
            foldername="$path/$name"
            for vers in "${CONDA_VERSIONS[@]}" ; do
                for bin in "${CONDA_BINS[@]}" ; do
                    condabin="$foldername$vers$bin"
                    if check_file_exists "$condabin" ; then
                        set_conda_dir_from_bin "$condabin"
                        CONDA_EXECUTABLE="$condabin";
                        retval=true
                        break 4
                    fi
                done
            done
        done
    done
    $retval
}

user_input() {
    # Get user options for install
    header "Welcome to the macOS Faceswap Installer"
    info "To get setup we need to gather some information about where you would like Faceswap\
    and Conda to be installed."
    info "To accept the default values just hit the 'ENTER' key for each option. You will have\
    an opportunity to review your responses prior to commencing the install."
    echo ""
    info "IMPORTANT: Make sure that the user '$USER' has full permissions for all of the\
    destinations that you select."
    read -rp $'\e[35m'"Press 'ENTER' to continue with the setup..."$'\e[39m'
    apps_opts
    conda_opts
    faceswap_opts
    post_install_opts
}

apps_opts () {
    # Options pertaining to additional apps that are required
    if ! command -V xquartz &> /dev/null ; then
        header "APPS"
        info "XQuartz is required to use the Faceswap GUI but was not detected. "
        if ask_yesno "Install XQuartz for GUI support?" "Yes" ; then
            XQUARTZ=true
        fi
    fi
}

conda_opts () {
    # Options pertaining to the installation of conda
    header "CONDA"
    info "Faceswap uses Conda as it handles the installation of all prerequisites."
    if find_conda_install && ask_yesno "Use the pre installed conda?" "Yes"; then
        info "Using Conda install at $DIR_CONDA"
    else
        echo ""
        info "If you have an existing Conda install then enter the location here,\
        otherwise Miniconda3 will be installed in the given location."
        err_msg="The location for Conda must not contain spaces (this is a specific\
        limitation of Conda)."
        tmp_dir_conda="$DIR_CONDA"
        while true ; do
            ask "Please specify a location for Conda." "DIR_CONDA"
            case ${DIR_CONDA} in
                *\ * ) error "$err_msg" ; DIR_CONDA=$tmp_dir_conda ;;
                * ) break ;;
            esac
        CONDA_EXECUTABLE="${DIR_CONDA}/bin/conda"
        done
    fi
    if ! check_file_exists "$CONDA_EXECUTABLE" ; then
        echo ""
        info "The Conda executable can be added to your PATH. This makes it easier to run Conda\
        commands directly. If you already have a pre-existing Conda install then you should\
        probably not enable this, otherwise this should be fine."
        if ask_yesno "Add Conda executable to path?" "Yes" ; then CONDA_TO_PATH=true ; fi
    fi
    echo ""
    info "Faceswap will be installed inside a Conda Environment. If an environment already\
    exists with the name specified then it will be deleted."
    ask "Please specify a name for the Faceswap Conda Environment" "ENV_NAME"
}

faceswap_opts () {
    # Options pertaining to the installation of faceswap
    header "FACESWAP"
    info "Faceswap will be installed in the given location. If a folder exists at the\
    location you specify, then it will be deleted."
    ask "Please specify a location for Faceswap" "DIR_FACESWAP"
    echo ""
    info "Faceswap can be run on Apple Silicon (M1, M2 etc.), compatible NVIDIA gpus, or on CPU. You should make sure that any \
    drivers are up to date. Please select the version of Faceswap you wish to install."
    ask_version
    if [ $VERSION == "apple_silicon" ] ; then
        DL_CONDA="${URL_CONDA}arm64.sh"
    fi
}

post_install_opts() {
    # Post installation options
    header "POST INSTALLATION ACTIONS"
    info "Launching Faceswap requires activating your Conda Environment and then running\
    Faceswap. The installer can simplify this by creating an Application Launcher file and placing it \
    on your desktop to launch straight into the Faceswap GUI"
    if ask_yesno "Create FaceswapGUI Launcher?" "Yes" ; then
        DESKTOP=true
    fi
}

review() {
    # Review user options and ask continue
    header "Review install options"
    info "Please review the selected installation options before proceeding:"
    echo ""
    if $XQUARTZ ; then echo "        - The XQuartz installer will be downloaded and launched" ; fi
    if ! check_folder_exists "$DIR_CONDA"
        then
            echo "        - MiniConda3 will be installed in '$DIR_CONDA'"
        else
            echo "        - Existing Conda install at '$DIR_CONDA' will be used"
    fi
    if $CONDA_TO_PATH ; then echo "        - MiniConda3 will be added to your PATH" ; fi
    if check_env_exists ; then
        echo $'        \e[33m- Existing Conda Environment '$ENV_NAME $' will be removed\e[39m'
    fi
    echo "        - Conda Environment '$ENV_NAME' will be created."
    if check_folder_exists "$DIR_FACESWAP" ; then
        echo $'        \e[33m- Existing Faceswap folder '$DIR_FACESWAP $' will be removed\e[39m'
    fi
    echo "        - Faceswap will be installed in '$DIR_FACESWAP'"
    echo "        - Installing for '$VERSION'"
    if [ $VERSION == "nvidia" ] ; then
        echo $'          \e[33m- Note: Please ensure that Nvidia drivers are installed prior to proceeding\e[39m'
    fi
    if $DESKTOP ; then echo "        - An Application Launcher will be created" ; fi
    if ! ask_yesno "Do you wish to continue?" "No" ;  then exit ; fi
}

xquartz_install() {
    # Download and install XQuartz
    if $XQUARTZ ; then
        info "Downloading XQuartz..."
        yellow ; download_file $DL_XQUARTZ
        echo ""

        info "Installing XQuartz..."
        info "Admin password required to install XQuartz:"
        fname="$(basename -- $DL_XQUARTZ)"
        yellow ; sudo installer -pkg "$TMP_DIR/$fname" -target /
        echo ""
    fi
}

conda_install() {
    # Download and install Mini Conda3
    if ! check_folder_exists "$DIR_CONDA" ; then
        info "Downloading Miniconda3..."
        yellow ; download_file $DL_CONDA
        info "Installing Miniconda3..."
        yellow ; fname="$(basename -- $DL_CONDA)"
        bash "$TMP_DIR/$fname" -b -p "$DIR_CONDA"
        if $CONDA_TO_PATH ; then
            info "Adding Miniconda3 to PATH..."
            yellow ; "$CONDA_EXECUTABLE" init zsh bash
            "$CONDA_EXECUTABLE" config --set auto_activate_base false
        fi
    fi
}

check_env_exists() {
    # Check if an environment with the given name exists
    if check_file_exists "$CONDA_EXECUTABLE" ; then
        "$CONDA_EXECUTABLE" env list | grep -qE "^${ENV_NAME}\W"
    else false
    fi
}

delete_env() {
    # Delete the env if it previously exists
    if check_env_exists ; then
        info "Removing pre-existing Virtual Environment"
        yellow ; "$CONDA_EXECUTABLE" env remove -n "$ENV_NAME"
    fi
}

create_env() {
    # Create Python 3.10 env for faceswap
    delete_env
    info "Creating Conda Virtual Environment..."
    yellow ; "$CONDA_EXECUTABLE" create -n "$ENV_NAME" -q python="$PYENV_VERSION" -y
}


activate_env() {
    # Activate the conda environment
    # shellcheck source=/dev/null
    source "$DIR_CONDA/etc/profile.d/conda.sh" activate
    conda activate "$ENV_NAME"
}

delete_faceswap() {
    # Delete existing faceswap folder
    if check_folder_exists "$DIR_FACESWAP" ; then
        info "Removing Faceswap folder: '$DIR_FACESWAP'"
        rm -rf "$DIR_FACESWAP"
    fi
}

clone_faceswap() {
    # Clone the faceswap repo
    delete_faceswap
    info "Downloading Faceswap..."
    yellow ; git clone --depth 1 --no-single-branch "$DL_FACESWAP" "$DIR_FACESWAP"
}

setup_faceswap() {
    # Run faceswap setup script
    info "Setting up Faceswap..."
    python -u "$DIR_FACESWAP/setup.py" --installer --$VERSION
}

create_gui_launcher () {
    # Create a shortcut to launch into the GUI
    launcher="$DIR_FACESWAP/faceswap_gui_launcher.command"
    launch_script="#!/bin/bash\n"
    launch_script+="source \"$DIR_CONDA/etc/profile.d/conda.sh\" activate && \n"
    launch_script+="conda activate '$ENV_NAME' && \n"
    launch_script+="python \"$DIR_FACESWAP/faceswap.py\" gui"
    printf "$launch_script" > "$launcher"
    chmod +x "$launcher"
}

create_app_on_desktop () {
    # Create a simple .app wrapper to launch GUI
    if $DESKTOP ; then
        app_name="FaceswapGUI"
        app_dir="$TMP_DIR/$app_name.app"

        unzip -qq "$DIR_FACESWAP/.install/macos/app.zip" -d "$TMP_DIR"

        script="#!/bin/bash\n"
        script+="bash \"$DIR_FACESWAP/faceswap_gui_launcher.command\""
        printf "$script" > "$app_dir/Contents/Resources/script"
        chmod +x "$app_dir/Contents/Resources/script"

        rm -rf "$HOME/Desktop/$app_name.app"
        mv "$app_dir" "$HOME/Desktop"
    fi ;
}

check_for_sudo
check_for_curl
check_for_xcode
banner
user_input
review
create_tmp_dir
xquartz_install
conda_install
create_env
activate_env
clone_faceswap
setup_faceswap
create_gui_launcher
create_app_on_desktop
info "Faceswap installation is complete!"
if $CONDA_TO_PATH ; then
    info "You should close the terminal before proceeding" ; fi
if $DESKTOP ; then info "You can launch Faceswap from the icon on your desktop" ; fi
if $XQUARTZ ; then
    warn "XQuartz has been installed. You must log out and log in again to be able to use the GUI" ; fi
