#!/usr/bin/env bash
# Run in local Ubuntu WSL with the Insomnia SSH control socket.
set -euo pipefail
login=${1:?user@host}
sock=${2:?control socket}
remote=${3:?absolute campaign directory}
dest=${4:?local results directory}
[[ "$remote" =~ ^/[a-zA-Z0-9_./-]+$ ]]
umask 077
mkdir -p "$dest"
transport="ssh -S $sock -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=60"
while true; do
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    # Slurm's command line and the transfer share the same authenticated transport.
    state=$(ssh -S "$sock" -o BatchMode=yes -o ConnectTimeout=12 "$login" \
        "export SLURM_CONF=/etc/slurm/slurm.conf; cd '$remote' && ids=\$(cut -d' ' -f1 submissions.txt | paste -sd, -) && squeue --me -h -o '%i,%T,%M,%R,%j' && sacct -X -nP -j \"\$ids\" -o JobID,State,Elapsed,ExitCode")
    printf '%s\n' "$state"
    rsync -rt --partial -e "$transport" --exclude=workers --exclude='state.sqlite*' --exclude=src \
        --exclude='*.tgz' "$login:$remote/" "$dest/"
    printf '%s\n' "$state" > "$dest/latest-slurm.txt"
    if ! cut -d' ' -f1 "$dest/submissions.txt" | grep -qFxf - <(printf '%s\n' "$state" | awk -F, 'NF>=5 {print $1}'); then
        echo "No submitted controller remains in the live queue."
        break
    fi
    sleep 300
done
