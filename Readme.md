# Smart Semi-automatic Annotation


Python Version - 3.6
cv2 version 4.5.3 (pip3 install opencv-python)



How to setup? (Setup using anconda prompt)


1. Update environment path if needed
https://www.how2shout.com/how-to/how-to-stop-python-from-opening-the-microsoft-store.html
https://www.geeksforgeeks.org/how-to-setup-anaconda-path-to-environment-variable/

2. Open VSCode - terminal command promt
   Install Python Verion 3.6
   conda env list
   conda remove --name py36rnn --all
   conda create --name py36rnn -c anaconda python=3.6
   conda activate py36rnn


3. Check python version in terminal
   python --version

4. pip upgrade
   C:\Users\salmank\anaconda3\envs\py36rnn\python.exe -m pip install pip==21.3.1

5. Install all packages as detailed in the requirements.txt
   set PIP_DEFAULT_TIMEOUT=1200

   pip install -r requirements.txt

6. open-cv installation
   pip install opencv-python==4.5.3.56
    
7. Copy Yolo weights from https://pjreddie.com/media/files/yolov3.weights to yolo-coco Directory

8. Clone this repo and rename folder as polyrnn https://github.com/fidler-lab/polyrnn-pp/tree/master

9. Download and extract following file to models folder (in plyrnn) http://www.cs.toronto.edu/polyrnn/models/checkpoints_cityscapes.tar.gz
   tar -xvf checkpoints_cityscapes.tar.gz ./polyrnn/models/

10. From terminal call:
    python yolo_click_crop_rnn2.py --image 'Image' --yolo yolo-coco

    Example:
    python yolo_click_crop_rnn2.py --image image_0.jpg --yolo yolo-coco
    python yolo_click_crop_rnn3.py --image g--yolo yolo-coco

3. Double click to crop and crop image is saved in the same directory.
