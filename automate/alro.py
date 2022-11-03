# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:40:37 2022

@author: LDE
"""

PROGRAM Main
// Version 0.11
VAR
	iStep : 						INT 	:= 0;				// Step du séquenceur
	// Interface HMI
	HMIOutputs : 					Outputs4HMI;
    HMIInputs : 					Inputs4HMI;
	// Interface Capsuleuse
	bCapsCheckLot1 		AT%I* : 	BOOL;
	bCapsCheckLot2 		AT%I* : 	BOOL;
	bCapsVisionReady 	AT%Q* : 	BOOL;
	bCapsLotOK 			AT%Q* : 	BOOL;
	bCapsDataValid 		AT%Q* : 	BOOL;
	bCapsCheckLot 		 	  : 	BOOL;
	// Interface caméra/Eclairage
	bLightOn 			AT%Q* : 	BOOL;
	bGrabImage 			AT%Q* : 	BOOL;
	bLightIsOn : 					BOOL;
	bDummyLightOn : 				BOOL;
	// Timers	
	tRunningTimer : 				TON;
	tCycleTime :					TIME;
	tFirstImage : 					TON;
	tSecondImage : 					TON;
	tFirstImageDelay : 				TIME 	:= T#190MS;
	tSecondImageDelay : 			TIME 	:= T#70MS;
	tLightOn : 						TON;
	tLightPreDelay : 				TIME 	:= T#10MS;
	tLightOff : 					TON;
	tLightPostDelay : 				TIME 	:= T#10MS;
	// Divers
	bWaitFirstImage : 				BOOL;
	bWaitSecondImage : 				BOOL;
	// Simulation de la capsuleuse
	bTest : 						BOOL 	:= FALSE;	
	bSimulHMI :						BOOL	:= FALSE;	
	tCycle : 						TON;
	bCountersResetted: BOOL;
END_VAR
VAR PERSISTENT
	dwCyclesCounter :				DWORD;
	dwGoodCounter :					DWORD;
	dwBadPrintCounter :				DWORD;
	dwBadColorCounter :				DWORD;
END_VAR



// Sélection du départ d'impression
bCapsCheckLot := bCapsCheckLot2;

// Temps de cycle
tRunningTimer(IN := (iStep <> 0),PT := T#60S);
tCycle(IN := TRUE,PT := T#610MS);

// Temporisation éclairage
tLightOn(IN := bDummyLightOn,PT := tLightPreDelay);
IF tLightOn.Q THEN
	bLightIsOn := TRUE;
END_IF;
tLightOff(IN := bLightIsOn,PT := tLightPreDelay);
IF tLightOff.Q THEN
	bDummyLightOn := FALSE;
	bLightIsOn := FALSE;
END_IF;

// Temporisation pour première image
tFirstImage(IN := bWaitFirstImage,PT := tFirstImageDelay-tLightPreDelay);
IF tFirstImage.Q THEN
	bDummyLightOn := TRUE;
	IF bLightIsOn THEN
		bWaitFirstImage := FALSE;
		bWaitSecondImage := TRUE;
	END_IF;
END_IF
IF bWaitFirstImage THEN
	HMIOutputs.bGet1stImage := TRUE;
END_IF;

// Temporisation pour 2e image
tSecondImage(IN := bWaitSecondImage,PT := tSecondImageDelay-tLightPreDelay);
IF tSecondImage.Q THEN
	bDummyLightOn := TRUE;
	IF bLightIsOn THEN
		bWaitSecondImage := FALSE;
	END_IF;
END_IF
IF bWaitSecondImage THEN
	HMIOutputs.bGet2ndImage := TRUE;
END_IF;

// On met le signal de start caméra quand la lumière est active
bGrabImage := bLightIsOn;

// Séquenceur
CASE iStep OF
	0: // Attente du start vision
		IF bTest AND tCycle.Q THEN
			tCycle(IN := FALSE);
			bCapsCheckLot := TRUE;
		END_IF
		IF bCapsCheckLot and not HMIInputs.bAnalyseStarted THEN
			HMIOutputs.bStart := TRUE;
			bWaitFirstImage := TRUE;
			iStep := 10;
		END_IF
	10:	// Attente de réponse du HMI
		IF bSimulHMI OR HMIInputs.bAnalyseStarted THEN
			HMIOutputs.bStart := FALSE;
			iStep := 15;
		END_IF
	15:	// Attente de réponse du HMI
		IF bSimulHMI OR NOT HMIInputs.bResultValid THEN
			iStep := 20;
		END_IF
	20:	// Attente du résultat d'analyse
		IF bSimulHMI or HMIInputs.bResultValid THEN
			dwCyclesCounter := dwCyclesCounter + 1;
			IF NOT HMIInputs.bPrintCheckOK THEN
				dwBadPrintCounter := dwBadPrintCounter + 1;
			ELSIF NOT HMIInputs.bCapsColorOK THEN
				dwBadColorCounter := dwBadColorCounter + 1;
			ELSE
				dwGoodCounter := dwGoodCounter + 1;
			END_IF
			HMIOutputs.bGet1stImage := FALSE;
			HMIOutputs.bGet2ndImage := FALSE;
			iStep := 30;
			tCycleTime := tRunningTimer.ET;
		END_IF
	30: // Fin cycle cpasuleuse
		IF bTest THEN
			bCapsCheckLot := FALSE;
			bTest := FALSE;
		END_IF
	 	IF NOT bCapsCheckLot THEN
			iStep := 0;
		END_IF
END_CASE

// Transfert vers la capsuleuse
bCapsVisionReady := HMIInputs.bReady;
bCapsDataValid := HMIInputs.bResultValid;
bCapsLotOK := HMIInputs.bPrintCheckOK AND HMIInputs.bCapsColorOK AND bCapsDataValid;
//bCapsLotOK := bCapsDataValid;

// Demande de remize à 0 des compteurs
IF HMIInputs.bResetCounters THEN
	dwGoodCounter := 0;
	dwBadPrintCounter := 0;
	dwBadColorCounter := 0;
	HMIOutputs.bCountersResetted := TRUE;
ELSE
	HMIOutputs.bCountersResetted := FALSE;
END_IF

// Recopie des compteurs pour le HMI
HMIOutputs.dwCntGood := dwGoodCounter;
HMIOutputs.dwCntBadPrint := dwBadPrintCounter;
HMIOutputs.dwCntBadCaps := dwBadColorCounter;

bLightOn := bDummyLightOn;
