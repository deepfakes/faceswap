#!/bin/bash

TMP_DIR="/tmp/faceswap_install"
DL_CONDA="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
DL_FACESWAP="https://github.com/deepfakes/faceswap.git"

CONDA_PATHS=("/opt" "$HOME")
CONDA_NAMES=("/ana" "/mini")
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

header() {
    # Format header text
    length=${#1}
    padding=$(( (72 - length) / 2))
    sep=$(printf '=%.0s' $(seq 1 $padding))
    echo ""
    echo -e "\e[32m$sep $1 $sep"
}

info () {
    # output info message
    while read -r line ; do
        echo -e "\e[32mINFO\e[97m    $line"
    done <<< "$(echo "$1" | fmt -cu -w 70)"
}

warn () {
    # output warning message
    while read -r line ; do
        echo -e "\e[33mWARNING\e[97m $line"
    done <<< "$(echo "$1" | fmt -cu -w 70)"
}

error () {
    # output error message.
    while read -r line ; do
        echo -e "\e[31mERROR\e[97m   $line"
    done <<< "$(echo "$1" | fmt -cu -w 70)"
}

yellow () {
    # Change text color to yellow
    echo -en "\e[33m"
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
    curl "$1" --output "$TMP_DIR/$fname" --progress-bar
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
        Please install 'curl' using the package manager for your distribution before proceeding."
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
    read -rp $'\e[36m'"$1 [default: '$default']: "$'\e[97m' inp
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
        read -rp $'\e[36m'"$1 $opts: "$'\e[97m' yn
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
        read -rp $'\e[36mSelect:\t1: NVIDIA\n\t2: AMD (ROCm)\n\t3: CPU\n'"[default: $default]: "$'\e[97m' vers
        vers="${vers:-${default}}"
        case $vers in
            1) VERSION="nvidia" ; break ;;
            2) VERSION="rocm" ; break ;;
            3) VERSION="cpu" ; break ;;
            * ) echo "Invalid selection." ;;
        esac
    done
}

banner () {
    echo -e "      \e[32m                   001"
    echo -e "      \e[32m                  11 10  010"
    echo -e "      \e[97m         @@@@\e[32m      10"
    echo -e "      \e[97m      @@@@@@@@\e[32m         00     1"
    echo -e "      \e[97m    @@@@@@@@@@\e[32m  1  1            0"
    echo -e "      \e[97m  @@@@@@@@\e[32m    0000 01111"
    echo -e "      \e[97m @@@@@@@@@@\e[32m    01  110 01  1"
    echo -e "      \e[97m@@@@@@@@@@@@\e[32m 111    010    0"
    echo -e "      \e[97m@@@@@@@@@@@@@@@@\e[32m  10    0"
    echo -e "      \e[97m@@@@@@@@@@@@@\e[32m   0010   1"
    echo -e "      \e[97m@@@@@@@@@  @@@\e[32m   100         1"
    echo -e "      \e[97m@@@@@@@ .@@@@\e[32m  10       1"
    echo -e "      \e[97m #@@@@@@@@@@@\e[32m  001       0"
    echo -e "      \e[97m   @@@@@@@@@@@  ,"
    echo -e "      \e[97m   @@@@@@@@  @@@@@"
    echo -e "      \e[97m  @@@@@@@@ @@@@@@@@"
    echo -e "      \e[97m @@@@@@@@@,@@@@@@@@  / _|"
    echo -e "      \e[97m %@@@@@@@@@@@@@@@@@ | |_  ___ "
    echo -e "      \e[97m     @@@@@@@@@@@@@@ |  _|/ __|"
    echo -e "      \e[97m      @@@@@@@@@@@@  | |  \__ \\"
    echo -e "      \e[97m       @@@@@@@@@@(  |_|  |___/"
    echo -e "      \e[97m          @@@@@@"
    echo -e "      \e[97m           @@@@"
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
    DIR_CONDA=$(readlink -f "$(dirname "$1")/..")
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
            foldername="$path${name}conda"
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
    header "Welcome to the Linux Faceswap Installer"
    info "To get setup we need to gather some information about where you would like Faceswap\
    and Conda to be installed."
    info "To accept the default values just hit the 'ENTER' key for each option. You will have\
    an opportunity to review your responses prior to commencing the install."
    echo ""
    info "\e[33mIMPORTANT:\e[97m Make sure that the user '$USER' has full permissions for all of the\
    destinations that you select."
    read -rp $'\e[36m'"Press 'ENTER' to continue with the setup..."$'\e[36m'
    conda_opts
    faceswap_opts
    post_install_opts
}

conda_opts () {
    # Options pertaining to the installation of conda
    header "CONDA"
    info "Faceswap uses Conda as it handles the installation of all prerequisites."
    if find_conda_install && ask_yesno "Use the pre installed conda?" "Yes"; then
        info "Using Conda install at $DIR_CONDA"
    else
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
    info "Faceswap can be run on NVIDIA or AMD GPUs or on CPU. You should make sure that you have the \
    latest graphics card drivers installed from the relevant vendor. Please select the version\
    of Faceswap you wish to install."
    ask_version
    if [ $VERSION == "rocm" ] ; then
        warn "ROCm support is experimental. Please make sure that your GPU is supported by ROCm and that \
        ROCm has been installed on your system before proceeding. Installation instructions: \
        https://docs.amd.com/bundle/ROCm_Installation_Guidev5.0/page/Overview_of_ROCm_Installation_Methods.html"
        sleep 2
    fi
}

post_install_opts() {
    # Post installation options
    if check_folder_exists "$HOME/Desktop" ; then
        header "POST INSTALLATION ACTIONS"
        info "Launching Faceswap requires activating your Conda Environment and then running\
        Faceswap. The installer can simplify this by creating a desktop shortcut to launch\
        straight into the Faceswap GUI"
        if ask_yesno "Create Desktop Shortcut?" "Yes"
            then DESKTOP=true
        fi
    fi
}

review() {
    # Review user options and ask continue
    header "Review install options"
    info "Please review the selected installation options before proceeding:"
    echo ""
    if ! check_folder_exists "$DIR_CONDA"
        then
            echo "        - MiniConda3 will be installed in '$DIR_CONDA'"
        else
            echo "        - Existing Conda install at '$DIR_CONDA' will be used"
    fi
    if $CONDA_TO_PATH ; then echo "        - MiniConda3 will be added to your PATH" ; fi
    if check_env_exists ; then
        echo -e "        \e[33m- Existing Conda Environment '$ENV_NAME' will be removed\e[97m"
    fi
    echo "        - Conda Environment '$ENV_NAME' will be created."
    if check_folder_exists "$DIR_FACESWAP" ; then
        echo -e "        \e[33m- Existing Faceswap folder '$DIR_FACESWAP' will be removed\e[97m"
    fi
    echo "        - Faceswap will be installed in '$DIR_FACESWAP'"
    echo "        - Installing for '$VERSION'"
    if [ $VERSION == "rocm" ] ; then
        echo -e "          \e[33m- Note: Please ensure that ROCm is supported by your GPU\e[97m"
        echo -e "          \e[33m  and is installed prior to proceeding.\e[97m"
    fi
    if $DESKTOP ; then echo "        - A Desktop shortcut will be created" ; fi
    if ! ask_yesno "Do you wish to continue?" "No" ;  then exit ; fi
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
            yellow ; "$CONDA_EXECUTABLE" init
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

install_git() {
    # Install git inside conda environment
    info "Installing Git..."
    # TODO On linux version 2.45.2 makes the font fixed TK pull in Python from
    # graalpy, which breaks pretty much everything
    yellow ; conda install "git<2.45" -q -y
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
    launcher="$DIR_FACESWAP/faceswap_gui_launcher.sh"
    launch_script="source \"$DIR_CONDA/etc/profile.d/conda.sh\" activate &&\n"
    launch_script+="conda activate '$ENV_NAME' &&\n"
    launch_script+="python \"$DIR_FACESWAP/faceswap.py\" gui\n"
    echo -e "$launch_script" > "$launcher"
    chmod +x "$launcher"
}

create_desktop_shortcut () {
    # Create a shell script to launch the GUI and add a desktop shortcut
    if $DESKTOP ; then
        desktop_icon="$HOME/Desktop/faceswap.desktop"
        desktop_file="[Desktop Entry]\n"
        desktop_file+="Version=1.0\n"
        desktop_file+="Type=Application\n"
        desktop_file+="Terminal=true\n"
        desktop_file+="Name=FaceSwap\n"
        desktop_file+="Exec=bash $launcher\n"
        desktop_file+="Comment=FaceSwap\n"
        desktop_file+="Icon=$DIR_FACESWAP/.install/linux/fs_logo.ico\n"
        echo -e "$desktop_file" > "$desktop_icon"
        chmod +x "$desktop_icon"
    fi ;
}

check_for_sudo
check_for_curl
banner
user_input
review
create_tmp_dir
conda_install
create_env
activate_env
install_git
clone_faceswap
setup_faceswap
create_gui_launcher
create_desktop_shortcut
info "Faceswap installation is complete!"
if $DESKTOP ; then info "You can launch Faceswap from the icon on your desktop" ; exit ; fi
if $CONDA_TO_PATH ; then
    info "You should close the terminal and re-open to activate Conda before proceeding" ; fi
