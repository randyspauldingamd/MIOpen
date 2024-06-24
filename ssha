#!/bin/bash
# Runs a new ssh-agent and loads the user's key for github

if [[ "${BASH_SOURCE[0]}" != "${0}" ]] ; then
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/github_ecdsa
else
  echo 'Error: script must be sourced'
fi
