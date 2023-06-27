!include MUI2.nsh
!include nsDialogs.nsh
!include winmessages.nsh
!include LogicLib.nsh
!include CPUFeatures.nsh
!include MultiDetailPrint.nsi

# Installer names and locations
OutFile "faceswap_setup_x64.exe"
Name "Faceswap"
InstallDir $PROFILE\faceswap

# Sometimes miniconda breaks. Uncomment/comment the following 2 lines to pin
!define wwwConda "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
#!define wwwConda "https://repo.anaconda.com/miniconda/Miniconda3-4.5.12-Windows-x86_64.exe"
!define wwwRepo "https://github.com/deepfakes/faceswap.git"
!define wwwFaceswap "https://www.faceswap.dev"

# Faceswap Specific
!define flagsSetup "--installer"

# Install cli flags
!define flagsConda "/S /RegisterPython=0 /AddToPath=0 /D=$PROFILE\MiniConda3"
!define flagsRepo "--depth 1 --no-single-branch ${wwwRepo}"
!define flagsEnv "-y python=3.10"

# Folders
Var ProgramData
Var dirTemp
Var dirMiniconda
Var dirMinicondaAll
Var dirAnaconda
Var dirAnacondaAll
Var dirConda

# Items to Install
Var InstallConda

# Misc
Var InstallFailed
Var lblPos
Var hasAVX
Var hasSSE4
Var setupType
Var ctlRadio
Var ctlCondaText
Var ctlCondaButton
Var Log
Var envName

# Modern UI2
!define MUI_COMPONENTSPAGE_NODESC
!define MUI_ABORTWARNING

# Install Location Page
!define MUI_ICON "fs_logo.ico"
!define MUI_PAGE_HEADER_TEXT "Faceswap.py Installer"
!define MUI_PAGE_HEADER_SUBTEXT "Install Location"
!define MUI_DIRECTORYPAGE_TEXT_DESTINATION "Select Destination Folder:"
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE VerifyInstallDir
!insertmacro MUI_PAGE_DIRECTORY

# Install Prerequisites Page
Page custom pgPrereqCreate pgPrereqLeave

# Install Faceswap Page
!define MUI_PAGE_CUSTOMFUNCTION_SHOW InstFilesShow
!define MUI_PAGE_HEADER_SUBTEXT "Installing Faceswap..."
!insertmacro MUI_PAGE_INSTFILES

# Set language (or Modern UI doesn't work)
!insertmacro MUI_LANGUAGE "English"

# Init
Function .onInit
    SetShellVarContext all
    StrCpy $ProgramData $APPDATA
    SetShellVarContext current
    # It's better to put stuff in $pluginsdir, $temp is shared
    InitPluginsDir
    StrCpy $dirTemp "$pluginsdir\faceswap\temp"
    StrCpy $dirMiniconda "$PROFILE\Miniconda3"
    StrCpy $dirAnaconda "$PROFILE\Anaconda3"
    StrCpy $dirMinicondaAll "$ProgramData\Miniconda3"
    StrCpy $dirAnacondaAll "$ProgramData\Anaconda3"
    StrCpy $envName "faceswap"
    SetOutPath "$dirTemp"
    Call CheckPrerequisites
FunctionEnd

# Enable the cancel button during installation
Function InstFilesShow
    GetDlgItem $0 $HWNDPARENT 2
    EnableWindow $0 1
FunctionEnd

Function VerifyInstallDir
    # Check install folder does not already exist
    IfFileExists  $INSTDIR 0 +3
    MessageBox MB_OK "Destination directory exists. Please select an alternative location"
    Abort
FunctionEnd

Function Abort
    MessageBox MB_OK "Some applications failed to install. Process Aborted. Check Details."
    Abort "Install Aborted"
FunctionEnd

Function pgPrereqCreate
    !insertmacro  MUI_HEADER_TEXT "Faceswap.py Installer" "Customize Install"

    nsDialogs::Create 1018
    Pop $0

    ${If} $0 == error
        Abort
    ${EndIf}

    StrCpy $lblPos 14
    # Info Installing applications
    ${NSD_CreateGroupBox} 5% 5% 90% 35% "The following applications will be installed"
    Pop $0

        ${If} $InstallConda == 1
            ${NSD_CreateLabel} 10% $lblPos% 80% 14u "MiniConda 3"
            Pop $0
            intOp $lblPos $lblPos + 7
        ${EndIf}
        ${NSD_CreateLabel} 10% $lblPos% 80% 14u "Faceswap"
        Pop $0

        StrCpy $lblPos 46
    # Info Custom Options
    ${NSD_CreateGroupBox} 5% 40% 90% 60% "Custom Items"
    Pop $0
        ${NSD_CreateRadioButton} 10% $lblPos% 27% 11u "Setup for NVIDIA GPU"
            Pop $ctlRadio
		    ${NSD_AddStyle} $ctlRadio ${WS_GROUP}
            nsDialogs::SetUserData $ctlRadio "nvidia"
            ${NSD_OnClick} $ctlRadio RadioClick
        ${NSD_CreateRadioButton} 40% $lblPos% 25% 11u "Setup for DirectML"
            Pop $ctlRadio
            nsDialogs::SetUserData $ctlRadio "directml"
            ${NSD_OnClick} $ctlRadio RadioClick
        ${NSD_CreateRadioButton} 70% $lblPos% 20% 11u "Setup for CPU"
            Pop $ctlRadio
            nsDialogs::SetUserData $ctlRadio "cpu"
            ${NSD_OnClick} $ctlRadio RadioClick

        intOp $lblPos $lblPos + 10

        ${NSD_CreateLabel} 10% $lblPos% 80% 10u "Environment Name (NB: Existing envs with this name will be deleted):"
        pop $0
        intOp $lblPos $lblPos + 7
        ${NSD_CreateText} 10% $lblPos% 80% 11u "$envName"
        Pop $envName
        intOp $lblPos $lblPos + 11


        ${If} $InstallConda == 1
            ${NSD_CreateLabel} 10% $lblPos% 80% 18u "Conda is required but could not be detected. If you have Conda already installed specify the location below, otherwise leave blank:"
            Pop $0
            intOp $lblPos $lblPos + 13

            ${NSD_CreateText} 10% $lblPos% 73% 12u ""
            Pop $ctlCondaText

            ${NSD_CreateButton} 83% $lblPos% 7% 12u "..."
            Pop $ctlCondaButton
            ${NSD_OnClick} $ctlCondaButton fnc_hCtl_test_DirRequest1_Click
        ${EndIf}

    nsDialogs::Show
FunctionEnd

Function RadioClick
    Pop $R0
	nsDialogs::GetUserData $R0
    Pop $setupType
FunctionEnd

Function fnc_hCtl_test_DirRequest1_Click
	Pop $R0
	${If} $R0 == $ctlCondaButton
		${NSD_GetText} $ctlCondaText $R0
		nsDialogs::SelectFolderDialog /NOUNLOAD "" "$R0"
		Pop $R0
		${If} "$R0" != "error"
			${NSD_SetText} $ctlCondaText "$R0"
		${EndIf}
	${EndIf}
FunctionEnd

Function pgPrereqLeave
	call CheckSetupType
    Call CheckCustomCondaPath
    ${NSD_GetText} $envName $envName

FunctionEnd

Function CheckSetupType
    ${If} $setupType == ""
	    MessageBox MB_OK "Please specify whether to setup for Nvidia, DirectML or CPU."
	    Abort
	${EndIf}
    StrCpy $Log "$log(check) Setting up for: $setupType$\n"
FunctionEnd

Function CheckCustomCondaPath
    ${NSD_GetText} $ctlCondaText $2
    ${If} $2 != ""
        nsExec::ExecToStack "$\"$2\Scripts\conda.exe$\" -V"
        pop $0
        pop $1
        ${If} $0 == 0
            StrCpy $InstallConda 0
            StrCpy $dirConda "$2"
            StrCpy $Log "$log(check) Custom Conda found: $1$\n"
        ${Else}
            StrCpy $Log "$log(error) Custom Conda not found at: $2. Installing MiniConda$\n"
        ${EndIf}
    ${EndIf}
FunctionEnd

Function CheckConda
    # miniconda
    nsExec::ExecToStack "$\"$dirMiniconda\Scripts\conda.exe$\" -V"
    pop $0
    pop $1

    nsExec::ExecToStack "$\"$dirMinicondaAll\Scripts\conda.exe$\" -V"
    pop $2
    pop $3

    # anaconda
    nsExec::ExecToStack "$\"$dirAnaconda\Scripts\conda.exe$\" -V"
    pop $4
    pop $5

    nsExec::ExecToStack "$\"$dirAnacondaAll\Scripts\conda.exe$\" -V"
    pop $6
    pop $7

    ${If} $0 == 0
        StrCpy $dirConda "$dirMiniconda"
        StrCpy $Log "$log(check) MiniConda installed: $1"
    ${ElseIf} $2 == 0
        StrCpy $dirConda "$dirMinicondaAll"
        StrCpy $Log "$log(check) MiniConda installed: $3"
    ${ElseIf} $4 == 0
        StrCpy $dirConda "$dirAnaconda"
        StrCpy $Log "$log(check) AnaConda installed: $5"
    ${ElseIf} $6 == 0
        StrCpy $dirConda "$dirAnacondaAll"
        StrCpy $Log "$log(check) AnaConda installed: $7"
    ${EndIf}
FunctionEnd

Function CheckPrerequisites
    # Conda
    Call CheckConda
    Push $PROFILE
        Call CheckForSpaces
    Pop $R0
    # If spaces in user profile look for and install Conda in C:
    ${If} $dirConda == ""
    ${AndIf} $R0 != 0
        StrCpy $dirMiniconda "C:\Miniconda3"
        StrCpy $dirAnaconda "C:\Anaconda3"
        Call CheckConda
    ${EndIf}

    ${If} $dirConda == ""
        StrCpy $InstallConda 1
    ${EndIf}

    # CPU Capabilities
        ${If} ${CPUSupports} "AVX2"
        ${OrIf} ${CPUSupports} "AVX1"
            StrCpy $Log "$log(check) CPU Supports AVX Instructions$\n"
            StrCpy $hasAVX 1
        ${EndIf}
        ${If} ${CPUSupports} "SSE4.2"
        ${OrIf} ${CPUSupports} "SSE4"
            StrCpy $Log "$log(check) CPU Supports SSE4 Instructions$\n"
            StrCpy $hasSSE4 1
        ${EndIf}

    StrCpy $Log "$Log(check) Completed check for installed applications$\n"
FunctionEnd

Function CheckForSpaces
# Check a string for space (Used for defining MiniConda install Location)
    Exch $R0
    Push $R1
    Push $R2
    Push $R3
    StrCpy $R1 -1
    StrCpy $R3 $R0
    StrCpy $R0 0
        loop:
        StrCpy $R2 $R3 1 $R1
        IntOp $R1 $R1 - 1
        StrCmp $R2 "" done
        StrCmp $R2 " " 0 loop
        IntOp $R0 $R0 + 1
    Goto loop
    done:
    Pop $R3
    Pop $R2
    Pop $R1
    Exch $R0

FunctionEnd

Section Install
    Push $Log
    Call MultiDetailPrint
    Call InstallConda
    Call SetEnvironment
    Call InstallGit
    Call CloneRepo
    Call SetupFaceSwap
    Call AddGuiLauncher
    Call DesktopShortcut
    ExecShell "open" "${wwwFaceswap}"
    DetailPrint "Visit ${wwwFaceswap} for help and support."
SectionEnd

Function InstallConda
    ${If} $InstallConda == 1
        DetailPrint "Downloading Miniconda3..."
        inetc::get /caption "Downloading Miniconda3." /canceltext "Cancel" ${wwwConda} "Miniconda3.exe" /end
        Pop $0
        ${If} $0 == "OK"
            DetailPrint "Installing Miniconda3. This will take a few minutes..."
            SetDetailsPrint listonly
            ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$\"$dirTemp\Miniconda3.exe$\" ${flagsConda}"
            pop $0
            ExecDos::wait $0
            pop $0
            StrCpy $dirConda "$dirMiniconda"
            SetDetailsPrint both
            ${If} $0 != 0
                DetailPrint "Error Installing Miniconda3"
                StrCpy $InstallFailed 1
            ${EndIf}
        ${Else}
            DetailPrint "Error Downloading Miniconda3"
            StrCpy $InstallFailed 1
        ${EndIf}
    ${EndIf}

    ${If} $InstallFailed == 1
        Call Abort
    ${Else}
        DetailPrint "Miniconda3 installed."
    ${EndIf}
FunctionEnd

Function SetEnvironment
    DetailPrint "Initializing Conda..."
    SetDetailsPrint listonly
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$\"$dirConda\scripts\activate.bat$\" && conda update -y -n base -c defaults conda && conda deactivate"
    pop $0
    ExecDos::wait $0
    pop $0
    SetDetailsPrint both
    DetailPrint "Creating Conda Virtual Environment..."

    IfFileExists  "$dirConda\envs\$envName" DeleteEnv CreateEnv
        DeleteEnv:
            DetailPrint "Removing existing Conda Virtual Environment..."
            SetDetailsPrint listonly
            ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$\"$dirConda\scripts\activate.bat$\" && conda env remove -y -n $\"$envName$\" && conda deactivate"
            pop $0
            ExecDos::wait $0
            pop $0
            SetDetailsPrint both
            ${If} $0 != 0
                DetailPrint "Error deleting Conda Virtual Environment"
                Call Abort
            ${EndIf}

        # Often Conda won't actually remove the folder and some of it's contents which leads to permission problems later
        IfFileExists  "$dirConda\envs\$envName" DeleteFolder CreateEnv
            DeleteFolder:
                DetailPrint "Deleting stale Conda Virtual Environment files..."
                SetDetailsPrint listonly
                RMDir /r "$dirConda\envs\$envName"
                pop $0
                SetDetailsPrint both
                ${If} $0 != 0
                    DetailPrint "Error deleting Conda Virtual Environment Folder"
                    Call Abort
                ${EndIf}

    CreateEnv:
        SetDetailsPrint listonly
        StrCpy $0 "${flagsEnv}"
        ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$\"$dirConda\scripts\activate.bat$\" && conda create $0 -n  $\"$envName$\" && conda deactivate"
        pop $0
        ExecDos::wait $0
        pop $0
        SetDetailsPrint both
        ${If} $0 != 0
            DetailPrint "Error Creating Conda Virtual Environment"
            Call Abort
        ${EndIf}
FunctionEnd

Function InstallGit
    DetailPrint "Installing Git..."
    SetDetailsPrint listonly
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$\"$dirConda\scripts\activate.bat$\" && conda activate $\"$envName$\" && conda install git -y -q && conda deactivate"
    pop $0
    ExecDos::wait $0
    pop $0
    SetDetailsPrint both
    ${If} $0 != 0
        DetailPrint "Error Installing Git"
        StrCpy $InstallFailed 1
    ${EndIf}
FunctionEnd

Function CloneRepo
    DetailPrint "Downloading Faceswap..."
    SetDetailsPrint listonly
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$\"$dirConda\scripts\activate.bat$\" && conda activate $\"$envName$\" && git clone ${flagsRepo} $\"$INSTDIR$\" && conda deactivate"
    pop $0
    ExecDos::wait $0
    pop $0
    SetDetailsPrint both
    ${If} $0 != 0
        DetailPrint "Error Downloading Faceswap"
        Call Abort
    ${EndIf}
FunctionEnd

Function SetupFaceSwap
    DetailPrint "Setting up FaceSwap Environment... This may take a while"
    StrCpy $0 "${flagsSetup}"
    StrCpy $0 "$0 --$setupType"
    SetDetailsPrint listonly
    ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$\"$dirConda\scripts\activate.bat$\" && conda activate $\"$envName$\" && python -u $\"$INSTDIR\setup.py$\" $0 && conda deactivate"
    pop $0
    ExecDos::wait $0
    pop $0
    SetDetailsPrint both
    ${If} $0 != 0
        DetailPrint "Error Setting up Faceswap"
        Call Abort
    ${EndIf}
FunctionEnd

Function AddGuiLauncher
    DetailPrint "Creating GUI Launcher"
    SetOutPath "$INSTDIR"
    StrCpy $0 "faceswap_win_launcher.bat"
    FileOpen $9 "$INSTDIR\$0" w
    FileWrite $9 "$\"$dirConda\scripts\activate.bat$\" && conda activate $\"$envName$\" && python $\"$INSTDIR/faceswap.py$\" gui$\r$\n"
    FileClose $9
FunctionEnd

Function DesktopShortcut
    DetailPrint "Creating Desktop Shortcut"
    CreateShortCut "$DESKTOP\FaceSwap.lnk" "$\"$INSTDIR\$0$\"" "" "$INSTDIR\.install\windows\fs_logo.ico"
FunctionEnd