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

            cv.destroyAllWindows()

            img=cv.imread('my_pic2.jpg')


            grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            eye_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye.xml')
            eyerect = eye_cascade.detectMultiScale(grayimg)

            print(eyerect)

            if len(eyerect) > 0:
                for rect in eyerect:
                    cv.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 255, 255), thickness=-1)


            img=cv.resize(img,(1024,768))
            cv.imshow('FacePhoto',img)


            cv.imwrite('my_pic2.jpg', img)
            
            cv.waitKey(0)
            cv.destroyAllWindows()
