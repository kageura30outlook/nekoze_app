# What this app does
This is a python app that checks if you are nekoze(have a bad posture) or not. If the code thinks you are nekoze, the code will show an image alerting you that you are nekoze. After the alert comes up you have to make your posture better(straighten your back). After 2 seconds with a good posture, the image goes away.
# How to use the app step by step
After you clone the repository, type in uv sync in your terminal.
After that type in uv run main.py.
Once the code starts, the code will take a frame from your camera that gets the data needed for the checking(the code will wait 10 seconds before the capture). Make sure you are in a bad posture when you do this, this is going to be the borderline of your level of bad posture. Below or about this posture will warn you that you have a bad posture.
After the capture, the code starts. If you are in a bad posture an image comes up, or else nothing happens.