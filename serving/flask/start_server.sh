#!/bin/sh

# good tutorial for server setup 
# https://www.e-tinkers.com/2018/08/how-to-properly-host-flask-application-with-nginx-and-guincorn/

sudo service nginx start
gunicorn --bind=unix:/tmp/gunicorn.sock --workers=4 run_medqa:app
