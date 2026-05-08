import cv2,time,json

YAW,PITCH,ROLL=18,15,15
SMOOTH=.75
FONT=cv2.FONT_HERSHEY_SIMPLEX

class Tracker:
    def __init__(self):
        self.data=[]
        self.start=time.time()

    def add(self,y,p,r,s):
        self.data.append([y,p,r,s])

    def export(self):
        report={"duration":round(time.time()-self.start,2),"frames":len(self.data),"suspicious":sum(i[3] for i in self.data)}
        with open("report.json","w") as f:json.dump(report,f)
        print("Report Saved")

def smooth(o,n):return SMOOTH*o+(1-SMOOTH)*n

def txt(f,t,p,s=.5,c=(255,255,255)):cv2.putText(f,t,p,FONT,s,c,1,cv2.LINE_AA)

def analyze(y,p,r):
    ay,ap,ar=abs(y),abs(p),abs(r)
    if not(ay>YAW or ap>PITCH or ar>ROLL):return "NORMAL",False
    if ay>YAW and ap>PITCH:
        if y>0 and p>0:return "DOWN-RIGHT",True
        if y<0 and p>0:return "DOWN-LEFT",True
        if y>0 and p<0:return "UP-RIGHT",True
        return "UP-LEFT",True
    if ay>YAW:return("RIGHT"if y>0 else"LEFT"),True
    if ap>PITCH:return("DOWN"if p>0 else"UP"),True
    return("TILT-RIGHT"if r>0 else"TILT-LEFT"),True