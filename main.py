import cv2,mediapipe as mp,numpy as np,time
from tracker import Tracker,smooth,txt,analyze

cap=cv2.VideoCapture(0)
mesh=mp.solutions.face_mesh.FaceMesh(min_detection_confidence=.5,min_tracking_confidence=.5)
tracker=Tracker()
sy=sp=sr=0
pt=time.time()

while True:
    ok,frame=cap.read()
    if not ok:break
    frame=cv2.flip(frame,1)
    h,w=frame.shape[:2]
    ct=time.time()
    fps=1/(ct-pt+1e-6)
    pt=ct
    res=mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    if res.multi_face_landmarks:
        face=res.multi_face_landmarks[0]
        f2d,f3d=[],[]
        for i,lm in enumerate(face.landmark):
            if i in[33,263,1,61,291,199]:
                x,y=int(lm.x*w),int(lm.y*h)
                f2d.append([x,y])
                f3d.append([x,y,lm.z])
        f2d=np.array(f2d,dtype=np.float64)
        f3d=np.array(f3d,dtype=np.float64)
        cam=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=np.float32)
        _,rot,_=cv2.solvePnP(f3d,f2d,cam,np.zeros((4,1)))
        ang,*_=cv2.RQDecomp3x3(cv2.Rodrigues(rot)[0])
        sy=smooth(sy,-ang[1]*360)
        sp=smooth(sp,-ang[0]*360)
        sr=smooth(sr,ang[2]*360)
        status,suspicious=analyze(sy,sp,sr)
        tracker.add(sy,sp,sr,suspicious)
        txt(frame,f"Y:{int(sy)} P:{int(sp)} R:{int(sr)}",(10,25))
        if suspicious:
            txt(frame,"SUSPICIOUS",(10,55),.8,(0,0,255))
            txt(frame,f"DIR: {status}",(10,85),.6,(0,165,255))
        else:
            txt(frame,"NORMAL",(10,55),.8,(0,255,0))
        txt(frame,f"FPS:{int(fps)}",(w-100,25),.5)
    else:
        txt(frame,"NO FACE",(w//2-80,h//2),1,(0,0,255))
    txt(frame,"Q=Quit E=Export",(10,h-10),.4)
    cv2.imshow("Head Pose Detection",frame)
    k=cv2.waitKey(1)&0xFF
    if k==ord("q"):break
    elif k==ord("e"):tracker.export()

cap.release()
cv2.destroyAllWindows()
tracker.export()