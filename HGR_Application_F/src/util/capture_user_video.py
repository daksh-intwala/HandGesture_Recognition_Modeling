from flask import Flask, render_template, request
from IPython.display import HTML
import base64
from base64 import b64encode
import cv2
import js2py
from IPython.display import display, Javascript,HTML
# from google.colab.output import eval_js
from base64 import b64decode

camera=cv2.VideoCapture(0)

class CaptureVideo:
    def __init__(self, filename):
        self.filename = filename
    
    def record_video(self):
        js=Javascript("""
        async function recordVideo() {
        const options = { mimeType: "video/webm; codecs=vp9" };
        const div = document.createElement('div');
        const capture = document.createElement('button');
        const stopCapture = document.createElement("button");
        
        capture.textContent = "Start Recording";
        capture.style.background = "orange";
        capture.style.color = "white";

        stopCapture.textContent = "Stop Recording";
        stopCapture.style.background = "red";
        stopCapture.style.color = "white";
        div.appendChild(capture);

        const video = document.createElement('video');
        const recordingVid = document.createElement("video");
        video.style.display = 'block';

        const stream = await navigator.mediaDevices.getUserMedia({audio:true, video: true});
        
        let recorder = new MediaRecorder(stream, options);
        document.body.appendChild(div);
        div.appendChild(video);

        video.srcObject = stream;
        video.muted = true;

        await video.play();

        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

        await new Promise((resolve) => {
            capture.onclick = resolve;
        });
        recorder.start();
        capture.replaceWith(stopCapture);

        await new Promise((resolve) => stopCapture.onclick = resolve);
        recorder.stop();
        let recData = await new Promise((resolve) => recorder.ondataavailable = resolve);
        let arrBuff = await recData.data.arrayBuffer();
        
        // stop the stream and remove the video element
        stream.getVideoTracks()[0].stop();
        div.remove();

        let binaryString = "";
        let bytes = new Uint8Array(arrBuff);
        bytes.forEach((byte) => {
            binaryString += String.fromCharCode(byte);
        })
        return btoa(binaryString);
        }
    """)
        try:
            display(js)
            data=js2py.eval_js('recordVideo({})')
            binary=b64decode(data)
            with open(self.filename,"wb") as video_file:
                video_file.write(binary)
            print(f"Finished recording video at:{self.filename}")
        except Exception as err:
            print(str(err))


    def show_video(self, video_width = 600):
    
        video_file = open(self.filename, "r+b").read()

        video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
        return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")
    

if __name__=="__main__":
    capture=CaptureVideo('test.mp4')
    capture.record_video()
    capture.show_video()