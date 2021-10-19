import picamera
import picamera.array
import cv2 as cv

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution = (512,384)


        while True:
            camera.capture(stream,'bgr',use_video_port=True)
            grayimg = cv.cvtColor(stream.array,cv.COLOR_BGR2GRAY)

            face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            facerect = face_cascade.detectMultiScale(grayimg,scaleFactor=1.2,minNeighbors=2,minSize=(100,100))

            if len(facerect)>0:
                cv.imwrite('my_pic2.jpg',stream.array)
                break

            cv.imshow('camera',stream.array)

            stream.seek(0)
            stream.truncate()

            if cv.waitKey(1)>0:
                break

        #cv.destroyAllW:indows()
