#!/bin/sh

currentShellPath=$(cd "$(dirname "$0")"; pwd)
remotePath="/Volumes/10.10.31.16/AILabs/CosyVoice"
rsync -av "$currentShellPath/../" "$remotePath/" --exclude ".git" --exclude ".github" --exclude ".DS_Store"
