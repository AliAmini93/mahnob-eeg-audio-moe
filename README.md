# MAHNOB-HCI EEG+Audio Continuous Valence Code

This repository provides code for continuous valence prediction with EEG and self-supervised audio representations on the MAHNOB-HCI dataset.

## Overview

The released scripts cover audio window construction, SSLAM audio embedding extraction, EEG-only and audio-only ablations, and EEG+Audio mixture-of-experts evaluation under session-preserving and content-held-out protocols.

## Data and supervision

The audio dataset builder uses 10-second windows with a 0.25-second step, matching the 4 Hz continuous annotation grid. For each label index k, the target is the continuous valence value at the same label index.

## Loss and metrics

The training scripts optimize standard concordance correlation coefficient loss and report RMSE, PCC, and CCC.

## Configuration

Set MAHNOB_BASE_DIR to the local MAHNOB-HCI working directory. Optionally set SSLAM_SAVE_DIR for local SSLAM storage.

## Requirements

Install dependencies with pip install -r requirements.txt. The audio dataset builder requires ffmpeg for MoviePy.
