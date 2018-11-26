from peyetribe import EyeTribe
import time

tracker = EyeTribe(host="localhost", port=6555)
tracker.connect()
n = tracker.next()

print("eT;dT;aT;Fix;State;Rwx;Rwy;Avx;Avy;LRwx;LRwy;LAvx;LAvy;LPSz;LCx;LCy;RRwx;RRwy;RAvx;RAvy;RPSz;RCx;RCy")

tracker.pushmode()
count = 0
while count < 500:
    n = tracker.next()
    print(n.righteye.avg)
    # print(n.lefteye)
    count += 1

tracker.pullmode()

tracker.close()