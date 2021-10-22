import cv2,time,pandas
from datetime import datetime

fst_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])

vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
    check,frame =vid.read()
    status=0

    gray_img= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_img= cv2.GaussianBlur(gray_img,(21,21),0)
    
    if fst_frame is None:
        fst_frame=gray_img #to capture the very first frame and save it
        continue

    delta_frame= cv2.absdiff(fst_frame,gray_img)
    thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame,None,iterations=2)
    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        if cv2.contourArea(contour)<10000:
            continue
        status=1

        (x,y,w,h) =cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    status_list.append(status)

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    cv2.imshow("capture", gray_img)
    cv2.imshow("deltaframe",delta_frame) #to show the difference we just calculated. 
    cv2.imshow("threshold frame",thresh_frame)
    cv2.imshow("Coloframe",frame)

    key = cv2.waitKey(1)#since the waitkey is not 0, it doesnt close on click of any key. 

    if key == ord('q'): #to quit window by pressing q
        if status==1:
            times.append(datetime.now())
        break
    
print(status_list)
print(times)

for i in range(0,len(times),2):
    df= df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)
df.to_csv("Times.csv")
vid.release()
cv2.destroyAllWindows()