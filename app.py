import os

import cv2
from flask import Flask, render_template, request, redirect, session, send_file, flash
from werkzeug.utils import secure_filename
import imports

app = Flask(__name__)
app.secret_key = 'sifeimeng'
files = []
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'wmv', 'avi', 'mkv'}


def is_allowed_file(filename):
    if '.' in filename:
        file_format = filename.rsplit('.')[1]
        if file_format in ALLOWED_EXTENSIONS:
            return True
    return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/infer', methods=['GET'])
def success():
    print("HERE")
    num_frames = session.get('num_frames', None)
    frame_size = session.get('frame_size', None)
    filename = files[-1]
    frame, score = imports.my_app(filename, num_frame=int(num_frames), frame_size=int(frame_size))
    cv2.imwrite('temp.jpg', frame)
    os.remove(filename)
    return render_template('inference.html', score=score, image=frame)


@app.route('/generate_image')
def generate_image():
    filename = 'temp.jpg'
    return send_file(filename, mimetype='image/jpg')


@app.route('/loading', methods=['GET', 'POST'])
def loading():
    if request.method == 'POST':
        f = request.files['file']
        num_frames = request.form.get('num_frames')
        frame_size = request.form.get('frame_size')
        session['num_frames'] = num_frames
        session['frame_size'] = frame_size
        if num_frames == "":
            flash("Number of frames not entered")
            return redirect('/')
        if frame_size == "":
            return redirect('/')
        if f.filename == "":
            return redirect('/')
        if is_allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(filename)
            files.append(filename)
        else:
            return redirect('/')
    return render_template('loading.html')


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)

