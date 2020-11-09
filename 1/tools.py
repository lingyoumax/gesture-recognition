import torch
import math
import numpy

def distinguish_rgb(data):
    length=data.shape[0]
    width=data.shape[1]
    for i in range(length):
        for j in range(width):
            R=data[i][j][0].item()
            G=data[i][j][1].item()
            B=data[i][j][2].item()
            if(R<95 or G<40 or B<20 or R<G or R<B or (max(R,max(G,B))-min(R,min(G,B)))<15 or abs(R-G)<15):
                data[i][j][0]=0
                data[i][j][1]=0
                data[i][j][2]=0   

    return data

def distinguish_yuv(data):
    length=data.shape[0]
    width=data.shape[1]
    for i in range(length):
        for j in range(width):
            R=data[i][j][0].item()
            G=data[i][j][1].item()
            B=data[i][j][2].item()
            Y = (65.481*R + 128.553*G + 24.966*B)/255+16
            U = (-81.085*R + 112*G -30.915*B)/255 + 128
            V = (112*R - 93.786*G - 18.214*B)/255 + 128
            if(U<77 or U>127 or V<133 or V>173):
                data[i][j][0]=0
                data[i][j][1]=0
                data[i][j][2]=0

    return data

def  wcb_y(y):
    y_min=16
    y_max=235
    kl=125
    kh=188
    wlcb=23
    whcb=14
    wcb=46.95
    w=0.0
    if y<kl:
        w+=wlcb+(y-y_min)*(wcb-wlcb)/(kl-y_min)
    if y>kh:
        w+=whcb+(y_max-y)*(wcb-whcb)/(y_max-kh)
    return w

def  wcr_y(y):
    y_min=16
    y_max=235
    kl=125
    kh=188
    wlcr=20
    whcr=10
    wcr=38.76
    w=0.0
    if y<kl:
        w+=wlcr+(y-y_min)*(wcr-wlcr)/(kl-y_min)
    if y>kh:
        w+=whcr+(y_max-y)*(wcr-whcr)/(y_max-kh)
    return w   

def cb_y(y):
    y_min=16
    y_max=235
    kl=125
    kh=188
    w=0.0
    if y<kl:
        w+=108+(kl-y)*(118-108)/(kl-y_min)
    if y>kh:
        w+=108+(y-kh)*(118-108)/(y_max-kh)
    return w

def cr_y(y):
    y_min=16
    y_max=235
    kl=125
    kh=188
    w=0.0
    if y<kl:
        w+=154+(kl-y)*(154-144)/(kl-y_min)
    if y>kh:
        w+=154+(y-kh)*(154-132)/(y_max-kh)
    return w

def c_b(y,cb):
    kl=125
    kh=188
    wcb=46.95
    if(y<kl or y>kh):
        cb=(cb-cb_y(y))*wcb/wcb_y(y)+cb_y(kh)
    return cb

def c_r(y,cr):
    kl=125
    kh=188
    wcr=38.76
    if(y<kl or y>kh):
        cr=(cr-cr_y(y))*wcr/wcr_y(y)+cr_y(kh)
    return cr 

def distinguish_ellipse(data):
    cx=109.38
    cy=152.02
    sita=2.53
    length=data.shape[0]
    width=data.shape[1]
    for i in range(length):
        for j in range(width):
            R=data[i][j][0].item()
            G=data[i][j][1].item()
            B=data[i][j][2].item()
            Y = (65.481*R + 128.553*G + 24.966*B)/255+16
            U = (-81.085*R + 112*G -30.915*B)/255 + 128
            V = (112*R - 93.786*G - 18.214*B)/255 + 128
            ctb=c_b(Y,U)-cx
            ctr=c_r(Y,V)-cy
            x=math.cos(sita)*ctb+math.sin(sita)*ctr
            y=math.cos(sita)*ctr-math.sin(sita)*ctb
            z=(x-1.60)**2/25.39**2+(y-2.41)**2/14.03**2
            if z>1:
                data[i][j][0]=0
                data[i][j][1]=0
                data[i][j][2]=0
    return data

def get_p(cb,cr):
    x=numpy.array([cb,cr])
    m=numpy.array([156.6699,117.4391])
    c=numpy.array([[299.457,12.1430],[12.1430,160.130]])
    c=numpy.linalg.inv(c)
    x=x-m
    p=numpy.dot(x,c)
    p=numpy.dot(p,x.T)
    p=-0.5*p
    p=numpy.exp(p)
    return p

def distinguish_gauss(data):
    length=data.shape[0]
    width=data.shape[1]
    for i in range(length):
        for j in range(width):
            R=data[i][j][0].item()
            G=data[i][j][1].item()
            B=data[i][j][2].item()
            Y = (65.481*R + 128.553*G + 24.966*B)/255+16
            U = (-81.085*R + 112*G -30.915*B)/255 + 128
            V = (112*R - 93.786*G - 18.214*B)/255 + 128
            p=get_p(U,V)
            if p>0.025:
                data[i][j][0]=0
                data[i][j][1]=0
                data[i][j][2]=0
    return data

def distinguish_otsu_gray_1(data):
    r=data[:,:,0].numpy()
    g=data[:,:,1].numpy()
    b=data[:,:,2].numpy()
    x=r*0.299 + g*0.587 + b*0.114
    x=x.reshape((1,-1))
    best_t=0
    best=0
    x=numpy.sort(x)
    x=x[0]
    num=x.shape[0]
    begin_index=0
    end_index=num-1
    while(x[begin_index]==x[0]):
        begin_index+=1

    while(x[end_index]==x[-1]):
        end_index-=1
    
    for i in range(begin_index,end_index+1,1):
        p0=float(i+1)/float(num)
        p1=1-p0
        u0=numpy.mean(x[0:i+1])
        u1=numpy.mean(x[i+1:num])
        variance=p0*p1*(u0-u1)*(u0-u1)
        if variance>best:
            best=variance
            best_t=x[i]
        
    length=data.shape[0]
    width=data.shape[1]
    for i in range(length):
        for j in range(width):
            R=data[i][j][0].item()
            G=data[i][j][1].item()
            B=data[i][j][2].item()
            V =R*0.299 + G*0.587 + B*0.114 
            if V<=best_t:
                data[i][j][0]=0
                data[i][j][1]=0
                data[i][j][2]=0        
    return data

def distinguish_otsu_gray_2(data):
    r=data[:,:,0].numpy()
    g=data[:,:,1].numpy()
    b=data[:,:,2].numpy()
    x=r*0.299 + g*0.587 + b*0.114
    x=x.reshape((1,-1))
    best_t=0
    best=0
    x=numpy.sort(x)
    x=x[0]
    num=x.shape[0]
    begin_index=0
    end_index=num-1
    while(x[begin_index]==x[0]):
        begin_index+=1

    while(x[end_index]==x[-1]):
        end_index-=1
    
    for i in range(begin_index,end_index+1,1):
        p0=float(i+1)/float(num)
        p1=1-p0
        u0=numpy.mean(x[0:i+1])
        u1=numpy.mean(x[i+1:num])
        variance=p0*p1*(u0-u1)*(u0-u1)
        if variance>best:
            best=variance
            best_t=numpy.mean(x[i:num])
        
    length=data.shape[0]
    width=data.shape[1]
    for i in range(length):
        for j in range(width):
            R=data[i][j][0].item()
            G=data[i][j][1].item()
            B=data[i][j][2].item()
            V =R*0.299 + G*0.587 + B*0.114 
            if V<=best_t:
                data[i][j][0]=0
                data[i][j][1]=0
                data[i][j][2]=0        
    return data              