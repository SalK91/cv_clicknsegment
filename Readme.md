# Smart Semi-automatic Annotation


Python Version - 3.6
cv2 version 4.5.3 (pip3 install opencv-python)



How to setup? (Setup using anconda prompt)


1. Update environment path if needed
https://www.how2shout.com/how-to/how-to-stop-python-from-opening-the-microsoft-store.html
https://www.geeksforgeeks.org/how-to-setup-anaconda-path-to-environment-variable/

2. Install Python Verion 3.6
   conda env list
   conda create --name py36 -c anaconda python=3.6
   conda activate py36


3. Check python version in terminal
   python --version

3. pip upgrade
   pip install --upgrade pip

4. open-cv installation
   pip install opencv-python==4.5.3.56


3. Copy Yolo weights from https://pjreddie.com/media/files/yolov3.weights to yolo-coco Directory

4. Clone this repo and rename folder as polyrnn
https://github.com/fidler-lab/polyrnn-pp/tree/master

2. Download and extract following file to models folder.
http://www.cs.toronto.edu/polyrnn/models/checkpoints_cityscapes.tar.gz

tar -xvf checkpoints_cityscapes.tar.gz ./polyrnn/models/


3. Install all packages as detailed in the requirements.txt
    pip install tensorflow
    pip install -r requirements.txt
3. From terminal call:
    python yolo_click_crop_rnn2.py --image 'Image' --yolo yolo-coco

    Example:
    python yolo_click_crop_rnn2.py --image image_0.jpg --yolo yolo-coco
    python yolo_click_crop_rnn2.py --image dog.jpg --yolo yolo-coco

3. Double click to crop and crop image is saved in the same directory.
