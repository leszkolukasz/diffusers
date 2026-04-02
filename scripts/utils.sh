#!/usr/bin/env bash


function rysy() {
    if ! ssh -O check icm;
    then
        ssh -N -f -M  icm
    fi
    ssh icm-rysy
}


function okeanos() {
    if ! ssh -O check icm;
    then
        ssh -N -f -M  icm
    fi
    ssh icm-okeanos
}


function mount-okeanos()
{
   MOUNTDIR="$HOME/Desktop/mountDir/OKEANOS"
   mkdir -p $MOUNTDIR
   sshfs $ICM_USERNAME@icm-okeanos:/lustre/tetyda/home/$ICM_USERNAME ${MOUNTDIR}
}

function mount-topola()
{
   MOUNTDIR="$HOME/Desktop/mountDir/TOPOLA"
   mkdir -p $MOUNTDIR
   sshfs $ICM_USERNAME@icm:/icm/home/$ICM_USERNAME ${MOUNTDIR}
}


function to_host {
    LOCAL_SOURCE=$1
    REMOTE_DESTINATION_DIR=$2
    # echo -e "Stuff to be copied to remote: $LOCAL_SOURCE \n"

    if ! test -z "$LOCAL_SOURCE" && ! test -z "$REMOTE_DESTINATION_DIR"
    then
        rsync -avzhe ssh --progress ${LOCAL_SOURCE} ${REMOTE_HOME_DIR}${REMOTE_DESTINATION_DIR}
    else
        echo "Usage: to_remote LOCAL_SOURCE REMOTE_DESTINATION_DIR"
    fi
}

function from_host {
  SOURCE_ON_REMOTE=$1
  LOCAL_DESTINATION_DIR=$2
  # echo -e "Stuff to be copied from remote: $SOURCE_ON_REMOTE \n"

  if ! test -z "$SOURCE_ON_REMOTE" && ! test -z "$LOCAL_DESTINATION_DIR"
  then
    rsync -avzhe ssh --progress ${REMOTE_HOME_DIR}${SOURCE_ON_REMOTE} ${LOCAL_DESTINATION_DIR}
  else
    echo "Usage: from_remote SOURCE_ON_REMOTE LOCAL_DESTINATION_DIR"
  fi
}

function to_icm {
    REMOTE_HOME_DIR="${ICM_USERNAME}@icm:/lu/topola/home/${ICM_USERNAME}/"
    to_host $@
}

function from_icm {
    REMOTE_HOME_DIR="${ICM_USERNAME}@icm:/lu/topola/home/${ICM_USERNAME}/"
    from_host $@
}

function to_okeanos {
    REMOTE_HOME_DIR="${ICM_USERNAME}@icm-okeanos:/lustre/tetyda/home/${ICM_USERNAME}/"
    to_host $@
}

function from_okeanos {
    REMOTE_HOME_DIR="${ICM_USERNAME}@icm-okeanos:/lustre/tetyda/home/${ICM_USERNAME}/"
    from_host $@
}

function to_rysy {
    REMOTE_HOME_DIR="${ICM_USERNAME}@icm-rysy:/home/${ICM_USERNAME}/"
    to_host $@
}

function from_rysy {
    REMOTE_HOME_DIR="${ICM_USERNAME}@icm-rysy:/home/${ICM_USERNAME}/"
    from_host $@
}