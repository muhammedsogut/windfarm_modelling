#!/bin/bash

for windir in {240..300}
do
    cd $windir'_degrees'
    cp *.csv ..
    cd ..
done
exit $exitcode
