import cv2
from keras.models import load_model
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

cascPath = 'haarcascade_files/haarcascade_frontalface_default.xml'
#model = load_model('vgg19.h5')
#model = load_model('our_resnet.h5')
#f = h5py.File('vgg19.h5', mode='r')

faceCascade = cv2.CascadeClassifier(cascPath)
model = load_model('sequential.h5')
print("model loaded ------------------------------------------------")

video_capture = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for face in faces:
        (x, y, w, h) = face
        frame = cv2.rectangle(frame, (x, y - 30), (x + w, y + h + 10), (255, 0, 0), 2)
        newimg = frame[y:y + h, x:x + w]
        newimg = cv2.resize(newimg, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.

        img = np.resize(newimg, [1, 48, 48, 1])

        result = model.predict(img)
        print("result:     ", result)
        rounded = [round(x[0]) for x in result]
        print(rounded)

        # Display the resulting frame
        #cv2.imshow('Video', frame)
        x, y = face[:2]
        if rounded == [1.0]: #happy
            cv2.putText(frame, "happy", (x , y ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,(255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "sad", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,(255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('Video', cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''
    print(
        "Happy: {0:.3f}%\nSad: {1:.3f}%\n",
        pred_array[0] * 100,
        pred_array[1] * 100
    )

    #plot
    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.225, 0.73, 0.15, 0.15]
    ax2 = fig.add_axes([left, bottom, width, height])

    plt.tick_params(
        axis='both',
        which='both',
        labelleft='on',
        labelbottom='off',
        top='off',
        bottom='off',
        left='off')
    ax1.tick_params(
        axis='both',
        which='both',
        labelleft='off',
        labelbottom='off',
        top='off',
        bottom='off',
        left='off')

    ax1.imshow(cropped, cmap="gray", vmin=0, vmax=255)

    Emotions = ['happy', 'sad']
    y_pos = np.arange(len(Emotions))
    ax2.barh(y_pos, pred_array, align='center', color='g')
    ax2.set_yticks(y_pos)
  #  ax2.set_yaicklabels(Emotions)
    ax2.patch.set_alpha(0.)  # transparent
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    plt.pause(1)  # random tiny timeout value to start loop over again


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
