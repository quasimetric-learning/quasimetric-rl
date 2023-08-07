#!/bin/bash

# default args are already for the online GCRL setting

exec python -m online.main env.kind=gcrl "${@}"
