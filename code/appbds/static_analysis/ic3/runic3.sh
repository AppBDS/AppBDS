#!/bin/sh

# Directory containing APK files (passed as the first argument)
appDir=$1

# Path to the Android platform jar (update this path as needed)
forceAndroidJar="path/to/android.jar"

var=1
for appPath in `ls $appDir/*.apk`
do
    appName=`basename $appPath .apk`
    retargetedPath="testspace/$appName/"
    echo "Processing $appName APK"

    mkdir -p output/$appName
    outputPath="output/$appName"

    # Run RetargetedApp.jar with a timeout of 18000 seconds and increased memory allocation
    gtimeout 18000 java -Xmx24000m -jar RetargetedApp.jar $forceAndroidJar $appPath $retargetedPath

    # Run ic3 analysis with a timeout of 18000 seconds and increased memory allocation
    gtimeout 18000 java -Xmx24000m -jar ic3-0.2.0-full.jar -apkormanifest $appPath -input $retargetedPath -cp $forceAndroidJar -db cc.properties -dbname cc1 -protobuf output2/$appName
    var=$((var+1))
done
echo "Finished processing $var APK files."
