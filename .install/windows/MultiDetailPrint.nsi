; Function:    MultiDetailPrint
; Written by:  M. Mims
; Date:        9.29.2006
; Description: Parses strings with return, newline, and tab characters to print
;              correctly to the details screen.
Function MultiDetailPrint
   Exch $R0
   Push $R1
   Push $R2
   Push $R3
   Push $R4
   StrLen $R1 $R0                      ; Get length of string
   StrCpy $R2 -1                       ; Set character index

 loop:
   IntOp $R2 $R2 + 1                   ; Increase character index
   StrCmp $R1 $R2 finish_trim          ; Finish if at end of string
   StrCpy $R3 $R0 1 $R2                ; Get character at given index
   StrCmp $R3 "$\r" r_trim_needed
   StrCmp $R3 "$\t" t_trim_needed
   StrCmp $R3 "$\n" n_trim_needed loop

 r_trim_needed:
   StrCpy $R3 $R0 $R2                  ; Copy left side of string
   IntOp $R4 $R2 + 1                   ; Get index of next character
   StrCpy $R0 $R0 $R1 $R4              ; Copy right side of string
   StrCpy $R0 "$R3$R0"                 ; Merge string without \r
   IntOp $R1 $R1 - 1                   ; Decrease string length
   IntOp $R2 $R2 - 1                   ; Decrease index
   goto loop

 t_trim_needed:
   StrCpy $R3 $R0 $R2                  ; Copy left side
   IntOp $R4 $R2 + 1                   ; Index of next character
   StrCpy $R0 $R0 $R1 $R4              ; Copy right side
   StrCpy $R0 "$R3        $R0"         ; Merge with spaces
   IntOp $R1 $R1 + 7                   ; Increase string length
   IntOp $R2 $R2 + 7                   ; Increase index
   goto loop

 n_trim_needed:
   StrCpy $R3 $R0 $R2                  ; Copy left side
   IntOp $R4 $R2 + 1                   ; Index of next character
   StrCpy $R0 $R0 $R1 $R4              ; Copy right side
   DetailPrint $R3                     ; Print line
   IntOp $R1 $R1 - $R2                 ; Adjust string length
   IntOp $R1 $R1 - 1
   StrCpy $R2 -1                       ; Adjust index
   goto loop

 finish_trim:
   DetailPrint $R0                     ; Print final line
   Pop $R4
   Pop $R3
   Pop $R2
   Pop $R1
   Exch $R0
FunctionEnd
