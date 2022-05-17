#!/usr/bin/python3

#NEED:
#1. enter core position for one map
#2. align images
#3. return flux at that position on both maps
#4. return spectral index at that position

"""

Last update : 2019 Feb 13
Current author contact information : Jae-Young Kim (jykim@mpifr-bonn.mpg.de)

Below is history of the program updates.

------------------

2019 Feb 13
v1.3

A large number of minor changes have been made in several places
to be compatible with recent versions of numpy and wxPython packages.

This version has been tested under the following environments:

Python 2.7.15
numpy 1.16.1
matplotlib 2.3.3
pyfits 3.5
wxPython 4.0.4

All the above versions are available from PyPI and the Conda python repository channel as of Feb 13, 2019.

Near future upgrade plan :
(1)  Replace the old pyfits package to the astropy fits io module,
(2)  Save the spectral index and error images in FITS format,
(3)  Support slicing by scipy image interpolation, similar to AIPS SLICE.

The program will also migrate to Github soon.

-------------------

2015 Mar 18
v1.2

Changed the SPIX map color.
A few lines are explicitly written to handle the rcParams.

-------------------

2014 Nov 28
v1.1

Fixed the colorbar scale in the last SPIX map procedure.
Also, fixed several minor problems.

-------------------

2014 Sep 04
v1.0

GUI-version of VLBI AGN IMage Alignment Program (or VIMAP).
Made by J.-Y. Kim at Seoul National University.
Current version and corresponding manual are available at
http://astro.snu.ac.kr/~trippe/VIMAP/vimap.html .

contact : jykim@astro.snu.ac.kr; trippe@astro.snu.ac.kr

"""
ver = 1.31

import wx
import os
import numpy as np
import astropy.io.fits as pf
import matplotlib
from glob import glob
matplotlib.use('WXAgg')
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_wxagg import \
                  FigureCanvasWxAgg as FigCanvas, \
                  NavigationToolbar2WxAgg as NavigationToolbar
import matplotlib.artist as Art
import warnings
import matplotlib.pyplot as plt


plt.close('all')

"""
For those who are familiar with Matplotlib,
change the rcParams as you like if you want to change some of the plot styles
(e.g. font, ticks, etc)
"""

plt_params={'font.family':'serif',
        'font.size':12,
        'text.usetex':'True'}
plt.rcParams.update(plt_params)


class TopFrame(wx.Frame):


    raw_img1   =0
    raw_img2   =0
    masked_img1=0
    masked_img2=0
    corr_center=0
    shift_x    =0
    shift_y    =0
    freq1      =0
    freq2      =0
    dxmas             =0
    dymas             =0
    dxmas2 = 0
    dymas2 = 0
    spix_err_limit = 0.5
    spix_cut   =0.1
    filepath=0
    re_xmin = 0
    re_xmax = 0
    re_ymin = 0
    re_ymax = 0
    mx1=0
    my1=0
    mx2=0
    my2=0
    ma1=0
    mb1=0
    mt1=0
    ma2=0
    mb2=0
    mt2=0
    cor_val=0
    shift_mas=0
    bmaj1 = 0
    bmin1 = 0
    bmaj2 = 0
    bmin2 = 0



    def __init__(self, parent, id, title):

        wx.Frame.__init__(self, parent, id, title, wx.DefaultPosition, wx.Size(700,80))

        panel1 = wx.Panel(self, -1)

        box= wx.BoxSizer(wx.HORIZONTAL)
        box.Add(wx.Button(panel1, 1, 'Load FITS'),1,wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, 1)
        box.Add(wx.Button(panel1, 2, 'Masking'),1,wx.ALIGN_CENTER | wx.EXPAND | wx.ALL,1)
        box.Add(wx.Button(panel1, 3, '2D Correlation'),1,wx.ALIGN_CENTER | wx.EXPAND | wx.ALL,1)
        box.Add(wx.Button(panel1, 4, 'SPIX map'),1,wx.ALIGN_CENTER | wx.EXPAND | wx.ALL,1)
        box.Add(wx.Button(panel1, 5, 'EXIT'),1,wx.ALIGN_CENTER | wx.EXPAND | wx.ALL,1)
        panel1.SetSizer(box)

        self.Bind(wx.EVT_BUTTON, self.OnFITS, id=1)
        self.Bind(wx.EVT_BUTTON, self.OnMASK, id=2)
        self.Bind(wx.EVT_BUTTON, self.OnRES, id=3)
        self.Bind(wx.EVT_BUTTON, self.OnSPIX, id=4)
        self.Bind(wx.EVT_BUTTON, self.OnEXIT, id=5)

        self.statusbar = self.CreateStatusBar()

        self.Centre()

        print("\n", "*"*20)
        print("Welcome to the VLBI IMage Alignment Program (VIMAP) ")
        print("*"*20, "\n")

        print("Step I. Load Images.\n")
        print("You have to")
        print("1) Load your map at low frequency, first.")
        print("2) Then, load high frequency map. \n")
        print("*"*20, "\n")


        self.path_list = []




    def OnFITS(self, event):
        self.statusbar.SetStatusText('Load two FITS Images from your local directory.')


        if len(self.path_list) > 2:
            self.path_list = []

        dlg = wx.FileDialog(self, "Choose a file", os.getcwd(), "", "*.*", wx.FD_MULTIPLE)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            mypath = os.path.basename(path)
            self.SetStatusText("You selected: %s" % mypath)

            if (mypath.split(".")[-1]).upper() == 'FITS':
                print("You selected FITS file at")
                print(path, "\n")

                if len(self.path_list)==0:
                    self.path_list.append(path)
                    print("#"*10)
                    print("Please load one more file.")
                    print("#"*10)
                    print('\n')

                elif len(self.path_list)==1:
                    self.path_list.append(path)
                    TopFrame.filepath = np.copy( self.path_list )
                    print("#"*10)
                    print("Now proceed to the Masking stage.")
                    print("#"*10)
                    print('\n')

                else:
                    self.path_list = []
                    print("!"*10)
                    print("You may selected more than two.")
                    print("Program resets your previous selections.")
                    print("Please load two FITS files again.")
                    print("!"*10)
                    print('\n')

            else:
                print("File extension is NOT FITS. Try correct files. \n")

        dlg.Destroy()

    def OnMASK(self, event):

        print("*"*20)
        print("\nStep II.")
        print("Entering Masking mode...\n")
        print("*"*20)


        self.statusbar.SetStatusText('Determine masking regions and image sizes for 2D Cross-correlation.')

        if len(self.path_list) == 2:
            self.mask_f= MaskFrame(None, -1, 'Masking Window', self.path_list)
            self.mask_f.Show()
        else:
            print("!"*10)
            print("Necessary FITS files are NOT read yet. Please load them first. \n")
            print("!"*10)

    def OnRES(self, event):


        self.statusbar.SetStatusText('Perform 2D Cross-Correlation using parameters determined from Masking task.')
        print("*"*20)
        print("Step III. Correlation.")
        print("*"*20)
        print("\nStart calculation ...\n")
        self.corr_f = CorrFrame(None, -1, '2D Cross-Correlation window')
        self.corr_f.Show()





    def OnSPIX(self, event):

        self.statusbar.SetStatusText('Examine Spectral Index map.')
        self.s_frame = SpixFrame(None, -1, 'Check spectral index distribution')
        self.s_frame.Show()




    def OnEXIT(self, event):
        print("\n")
        print(" Bye !")
        print("\n")
        try:
            self.mask_f.Close()
        except:
            pass
        try:
            self.corr_f.Close()
        except:
            pass
        try:
            self.s_frame.Close()
        except:
            pass
        self.Close()







class ImageDisp(wx.Panel):

    def __init__(self, parent,img_path):

        wx.Panel.__init__(self, parent)

        img_list = img_path

        self.low_map = pf.open(img_list[0])
        self.high_map = pf.open(img_list[1])

        self.img1 = self.low_map[0].data[0][0]
        self.img2 = self.high_map[0].data[0][0]
        self.img1_backup = self.img1[:,:]
        self.img2_backup = self.img2[:,:]

        TopFrame.raw_img1 = np.copy(self.img1)
        TopFrame.raw_img2 = np.copy(self.img2)

        xn1 = len( self.img1[0,:] )
        yn1 = len( self.img1[:,0] )
        xn2 = len( self.img2[0,:] )
        yn2 = len( self.img2[:,0] )

        if (xn1 != xn2 or yn1 != yn2):
            raise Exception("Numbers of pixels of the two FITS are not the same. Please check your data in advance.")

        head_info = self.low_map[0].header
        ra1_c  = head_info['CRVAL1']
        dec1_c = head_info['CRVAL2']
        dx     = head_info['CDELT1']
        dy     = head_info['CDELT2']
        x0     = head_info['CRPIX1']
        y0     = head_info['CRPIX2']
        TopFrame.bmaj1  = head_info['BMAJ'] *3600*1000
        TopFrame.bmin1  = head_info['BMIN'] *3600*1000

        head_info2 = self.high_map[0].header
        ra2_c   = head_info2['CRVAL1']
        dec2_c  = head_info2['CRVAL2']
        dx2     = head_info2['CDELT1']
        dy2     = head_info2['CDELT2']
        x02     = head_info2['CRPIX1']
        y02     = head_info2['CRPIX2']
        TopFrame.bmaj2   = head_info2['BMAJ'] *3600*1000
        TopFrame.bmin2   = head_info2['BMIN'] *3600*1000



        #print(self.img2[point_x, point_y])

        for i in range(1,10):
            try:
                test_head1 = head_info['CTYPE'+str(i)]
                test_head2 = head_info2['CTYPE'+str(i)]
                if test_head1.split()[0] == 'FREQ':
                    TopFrame.freq1= head_info['CRVAL'+str(i)]
                if test_head2.split()[0] == 'FREQ':
                    TopFrame.freq2= head_info2['CRVAL'+str(i)]
            except:
                pass

        TopFrame.dxmas = abs(dx)*3600*1000
        TopFrame.dymas = abs(dy)*3600*1000
        TopFrame.dxmas2 = abs(dx2)*3600*1000
        TopFrame.dymas2 = abs(dy2)*3600*1000

        self.img_num=True
        self.minimum_lev=1
        self.remap_xmin=0
        self.remap_xmax=0
        self.remap_ymin=0
        self.remap_ymax=0
        self.do_mask = 0
        self.mask_x1 = 250
        self.mask_y1 = 250
        self.mask_a1 = 30
        self.mask_b1 = 30
        self.mask_t1 = 0
        self.mask_x2 = 250
        self.mask_y2 = 250
        self.mask_a2 = 30
        self.mask_b2 = 30
        self.mask_t2 = 0


        self.init_plot()

        sizer= wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas,1,wx.ALL,0)
        self.SetSizer(sizer)

        self.Bind(wx.EVT_SIZE, self.sizeHandler)

    def init_plot(self):

        self.dpi = 100
        self.fig = Figure((4.,4.), dpi=self.dpi)
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.ml=MultipleLocator(5)

        self.settings()

        self.axes.set_title('Low frequency map', size=17)
        self.plot_data = self.axes.contour( self.img1, self.levels, colors='r', origin='lower')

        self.init_img_size = [ 1, min( len(self.img1[0,:]), len(self.img2[0,:]) ) , 1, min( len(self.img1[:,0]), len(self.img2[:,0]) ) ]

        self.axes.set_xlim([ 1,len( self.img1[0,:] ) ])
        self.axes.set_ylim([ 1,len( self.img2[:,0] ) ])

        self.canvas.draw()

    def replot(self):

        self.axes.clear()
        try:
            self.rep_img1=  np.copy( self.img1[ int(self.remap_ymin) -1 : int(self.remap_ymax) -1,
                                                                              int(self.remap_xmin) -1 : int(self.remap_xmax) -1] )
            self.rep_img2=  np.copy( self.img2[ int(self.remap_ymin) -1 : int(self.remap_ymax) -1,
                                                                              int(self.remap_xmin) -1 : int(self.remap_xmax) -1] )
        except:
            print("Image size makes trouble. Please check input values.")

        TopFrame.re_xmin = np.copy(self.remap_xmin)
        TopFrame.re_xmax = np.copy(self.remap_xmax)
        TopFrame.re_ymin = np.copy(self.remap_ymin)
        TopFrame.re_ymax = np.copy(self.remap_ymax)

        try:
            self.rep_xlim = np.arange( int(self.remap_xmin), int(self.remap_xmax), 1)
            self.rep_ylim = np.arange( int(self.remap_ymin), int(self.remap_ymax), 1)
        except:
            print("Please check resize parameters.")

        self.settings()
        if self.do_mask==1:
            self.PerformMasking()
            if self.img_num:
                self.plot_data = self.axes.contour( self.rep_xlim, self.rep_ylim, self.rep_img1, self.levels, colors='r', origin='lower' )
                self.axes.set_title('Low frequency map', size=17)
            else:
                self.plot_data = self.axes.contour( self.rep_xlim, self.rep_ylim, self.rep_img2, self.levels, colors='r', origin='lower' )
                self.axes.set_title('High frequency map', size=17)

        else:
            if self.img_num:
                self.plot_data = self.axes.contour( self.rep_xlim, self.rep_ylim, self.rep_img1, self.levels, colors='r', origin='lower' )
                self.axes.set_title('Low frequency map', size=17)
            else:
                self.plot_data = self.axes.contour( self.rep_xlim, self.rep_ylim, self.rep_img2, self.levels, colors='r', origin='lower' )
                self.axes.set_title('High frequency map', size=17)

            self.masked_img1 = np.copy( self.rep_img1 )
            self.masked_img2 = np.copy( self.rep_img2 )

        if self.img_num:
            self.plot_data = self.axes.contour( self.rep_xlim, self.rep_ylim, self.rep_img1, self.levels, colors='r', origin='lower' )
            self.axes.set_title('Low frequency map', size=17)
        else:
            self.plot_data = self.axes.contour( self.rep_xlim, self.rep_ylim, self.rep_img2, self.levels, colors='r', origin='lower' )
            self.axes.set_title('High frequency map', size=17)

        self.canvas.draw()

    def settings(self):

        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('white')

        Art.setp(self.axes.get_xticklabels(), fontsize=16)
        Art.setp(self.axes.get_yticklabels(), fontsize=16)

        self.axes.set_xlabel('relative RA [in Pixel]', fontsize=16)
        self.axes.set_ylabel('relative DEC [in Pixel]', fontsize=16)

        self.axes.xaxis.set_minor_locator(self.ml)
        self.axes.yaxis.set_minor_locator(self.ml)

        self.levels = 2. ** np.arange( np.log10(self.minimum_lev*1e-2*self.img1.max())/np.log10(2), np.log10(self.img1.max())/np.log10(2), 1.0)
        np.append( -1*2.**( np.log10(self.minimum_lev*1e-2*self.img1.max())/np.log10(2)  ) , self.levels)



    def PerformMasking(self):


        TopFrame.mx1 = np.copy(self.mask_x1)
        TopFrame.my1 = np.copy(self.mask_y1)
        TopFrame.mx2 = np.copy(self.mask_x2)
        TopFrame.my2 = np.copy(self.mask_y2)
        TopFrame.ma1 =  np.copy(self.mask_a1)
        TopFrame.mb1 = np.copy(self.mask_b1)
        TopFrame.mt1 = np.copy(self.mask_t1)
        TopFrame.ma2 =  np.copy(self.mask_a2)
        TopFrame.mb2 =  np.copy(self.mask_b2)
        TopFrame.mt2 =  np.copy(self.mask_t2)


        self.mask_x1 = np.int(self.mask_x1) - np.int(self.remap_xmin)
        self.mask_y1 = np.int(self.mask_y1) - np.int(self.remap_ymin)
        self.mask_x2 = np.int(self.mask_x2) - np.int(self.remap_xmin)
        self.mask_y2 = np.int(self.mask_y2) - np.int(self.remap_ymin)

        y1, x1 = np.ogrid[ -int(self.mask_y1):len(self.rep_img1[:,0]) -int(self.mask_y1) ,
                                                                                  -int(self.mask_x1):len(self.rep_img1[0,:]) -int(self.mask_x1) ]
        y2, x2 = np.ogrid[ -int(self.mask_y2):len(self.rep_img2[:,0]) -int(self.mask_y2),
                                                                                  -int(self.mask_x2):len(self.rep_img2[0,:]) -int(self.mask_x2) ]

        x1n = np.cos( np.float(self.mask_t1) )*(x1) - np.sin( np.float(self.mask_t1) )*(y1)
        y1n = np.sin( np.float(self.mask_t1) )*(x1) + np.cos( np.float(self.mask_t1) )*(y1)

        mask1 = ( (x1n)**2/np.float(self.mask_a1)**2  + (y1n)**2/np.float(self.mask_b1)**2 <= 1. )
        try:
            self.rep_img1[mask1]=0
            self.masked_img1 = np.copy( self.rep_img1 )
        except:
            print("Masking region (of img1) is too large. \n")


        x2n = np.cos( np.float(self.mask_t2) )*(x2) - np.sin( np.float(self.mask_t2) )*(y2)
        y2n = np.sin( np.float(self.mask_t2) )*(x2) + np.cos( np.float(self.mask_t2) )*(y2)

        mask2 = ( (x2n)**2/np.float(self.mask_a2)**2  + (y2n)**2/np.float(self.mask_b2)**2 <= 1. )
        try:
            self.rep_img2[mask2]=0
            self.masked_img2= np.copy( self.rep_img2 )
        except:
            print("Masking region (of img2) is too large. \n")

    def sizeHandler(self, *args, **kwargs):
        self.canvas.SetSize(self.GetSize())




class MaskFrame(wx.Frame):

    def __init__(self, parent, id, title, img_path):

        wx.Frame.__init__(self, parent, id, title, wx.DefaultPosition, wx.Size(1000,800))

        panel1 = wx.Panel(self, -1, style=wx.SUNKEN_BORDER)

        self.image_path = img_path
        self.maps = ImageDisp(panel1, self.image_path)

        sizer_h = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h.Add( self.maps , proportion = 1,  flag = wx.ALIGN_CENTER | wx.RIGHT | wx.EXPAND , border=0)



        sizer_v = wx.BoxSizer(wx.VERTICAL)


        self.init_mask_par = np.array( [250,250,30,30,0] , dtype='int')

        self.image_control = ImgBoxCtrl(panel1, -1, "Select Image", 10)
        self.size_setting = SizeSetting(panel1, -1, "Image resize params [Pixel]", self.maps.init_img_size)
        self.img1_xy_ctrl = SetMask(panel1, -1, "Mask Img1", self.init_mask_par)
        self.img2_xy_ctrl = SetMask(panel1, -1, "Mask Img2", self.init_mask_par)

        imchoice_box = wx.BoxSizer(wx.VERTICAL)
        imchoice_box.Add(self.image_control, 0, wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, 5)

        sizeset_box = wx.BoxSizer(wx.HORIZONTAL)
        sizeset_box.Add(self.size_setting, 0, wx.ALIGN_CENTER | wx.ALL | wx.EXPAND, 5)

        minlev_box = wx.BoxSizer(wx.VERTICAL)
        self.minlev_resize = wx.CheckBox(panel1, -1,
                        "Resize contours",
                        style=wx.ALIGN_RIGHT)
        minlev_box.Add(self.minlev_resize, 0, wx.ALIGN_CENTER | wx.ALL , 5)
        lev_text = wx.StaticText(panel1, -1, 'Minimum contour level [in 0.1%]', style = wx.ALIGN_CENTER)
        self.sld = wx.Slider(panel1, -1,
                        1,1,150, wx.DefaultPosition, (220,-1),
                        wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS, name='Minimum Level')
        self.sld.SetTickFreq(1)
        minlev_box.Add(lev_text, 0, wx.ALIGN_CENTER | wx.TOP, 1)
        minlev_box.Add(self.sld, 0, wx.ALIGN_CENTER | wx.BOTTOM, 1)

        resizing_box = wx.BoxSizer(wx.HORIZONTAL)
        self.cb_resize = wx.CheckBox(panel1, -1,
                        "Do Resizing",
                        style=wx.ALIGN_RIGHT)
        resizing_box.Add(self.cb_resize, 0, wx.ALIGN_CENTER | wx.ALL , 5)

        params_box = wx.BoxSizer(wx.HORIZONTAL)
        params_box.Add(self.img1_xy_ctrl, 0,  wx.ALIGN_CENTER | wx.ALL | wx.EXPAND, 5)
        params_box.Add(self.img2_xy_ctrl, 0,  wx.ALIGN_CENTER | wx.ALL | wx.EXPAND, 5)

        masking_box = wx.BoxSizer(wx.HORIZONTAL)
        self.cb_mask = wx.CheckBox(panel1, -1,
                        "Do Masking",
                        style=wx.ALIGN_RIGHT)
        masking_box.Add(self.cb_mask, 1, wx.ALIGN_CENTER | wx.ALL , 5)

        gs2 = wx.GridSizer(3,1,1,0)
        gs2.AddMany( [(wx.Button(panel1, 209,'PLOT'),1, wx.RIGHT | wx.LEFT | wx.ALIGN_CENTER | wx.EXPAND, 5 ),
                                          (wx.Button(panel1, 210,'DONE'),1, wx.RIGHT | wx.LEFT | wx.ALIGN_CENTER | wx.EXPAND, 5 ),
                                          (wx.Button(panel1, 299,'EXIT'),1, wx.RIGHT | wx.LEFT | wx.ALIGN_CENTER | wx.EXPAND, 5 )] )

        self.Bind(wx.EVT_BUTTON, self.OnEXIT, id=299)
        self.Bind(wx.EVT_BUTTON, self.OnDone, id=210)
        self.Bind(wx.EVT_BUTTON, self.OnMapSet, id=202)
        self.Bind(wx.EVT_BUTTON, self.Update, id=209)


        sizer_v.Add(imchoice_box, 0, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.TOP, 20)
        sizer_v.AddSpacer(20)
        sizer_v.Add(minlev_box, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, 5)
        sizer_v.Add(resizing_box, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, 5)
        sizer_v.Add(sizeset_box, 0, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)
        sizer_v.Add(masking_box, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, 5)
        sizer_v.Add(params_box, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, 5)
        sizer_v.Add(gs2, 0, wx.EXPAND | wx.ALIGN_BOTTOM | wx.ALIGN_CENTER | wx.ALL, border=10)

        sizer_h.Add(sizer_v, 0, wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, 1)

        panel1.SetSizer(sizer_h)


    def OnDone(self, event):
        try:
            TopFrame.masked_img1 = np.copy(self.maps.masked_img1)
            self.ok1=1
        except:
            print("LOW FREQ map is not checked yet.")
        try:
            TopFrame.masked_img2 = np.copy(self.maps.masked_img2)
            self.ok2=1
        except:
            print("HIGH FREQ map is not checked yet." )

        if (self.ok1==1 and self.ok2==1):
            print("Now proceed further to 2D Corr menu.")

    def OnMapSet(self, event):
        dlg = Misc(None, -1, "Map Settings")
        if dlg.ShowModal() == wx.ID_OK:
            self.new_value = dlg.GetVals()
            self.update_values(self.new_value)
        dlg.Destroy()

    def Update(self, event):

        self.maps.minimum_lev = 0.01*(self.sld.GetValue())

        if self.cb_resize.GetValue():
            self.maps.remap_xmin = self.size_setting.xmin.GetValue()
            self.maps.remap_xmax = self.size_setting.xmax.GetValue()
            self.maps.remap_ymin = self.size_setting.ymin.GetValue()
            self.maps.remap_ymax = self.size_setting.ymax.GetValue()
        else:
            self.maps.remap_xmin = self.maps.init_img_size[0]
            self.maps.remap_xmax = self.maps.init_img_size[1]
            self.maps.remap_ymin = self.maps.init_img_size[2]
            self.maps.remap_ymax = self.maps.init_img_size[3]


        if self.image_control.img_value():
            self.maps.img_num=1
            if self.cb_mask.GetValue():
                self.maps.do_mask = 1
                self.maps.mask_x1 = self.img1_xy_ctrl.xcenter.GetValue()
                self.maps.mask_y1 = self.img1_xy_ctrl.ycenter.GetValue()
                self.maps.mask_a1 = self.img1_xy_ctrl.major.GetValue()
                self.maps.mask_b1 = self.img1_xy_ctrl.minor.GetValue()
                self.maps.mask_t1 = -np.float(self.img1_xy_ctrl.tilt.GetValue())*np.pi/180
                self.maps.mask_x2 = self.img2_xy_ctrl.xcenter.GetValue()
                self.maps.mask_y2 = self.img2_xy_ctrl.ycenter.GetValue()
                self.maps.mask_a2 = self.img2_xy_ctrl.major.GetValue()
                self.maps.mask_b2 = self.img2_xy_ctrl.minor.GetValue()
                self.maps.mask_t2 = -np.float(self.img2_xy_ctrl.tilt.GetValue())*np.pi/180
            else:
                self.maps.do_mask = 0
        else:
            self.maps.img_num=0
            if self.cb_mask.GetValue():
                self.maps.do_mask = 1
                self.maps.mask_x1 = self.img1_xy_ctrl.xcenter.GetValue()
                self.maps.mask_y1 = self.img1_xy_ctrl.ycenter.GetValue()
                self.maps.mask_a1 = self.img1_xy_ctrl.major.GetValue()
                self.maps.mask_b1 = self.img1_xy_ctrl.minor.GetValue()
                self.maps.mask_t1 = -np.float(self.img1_xy_ctrl.tilt.GetValue())*np.pi/180
                self.maps.mask_x2 = self.img2_xy_ctrl.xcenter.GetValue()
                self.maps.mask_y2 = self.img2_xy_ctrl.ycenter.GetValue()
                self.maps.mask_a2 = self.img2_xy_ctrl.major.GetValue()
                self.maps.mask_b2 = self.img2_xy_ctrl.minor.GetValue()
                self.maps.mask_t2 = -np.float(self.img2_xy_ctrl.tilt.GetValue())*np.pi/180
            else:
                self.maps.do_mask = 0

        self.maps.replot()



    def OnEXIT(self, event):
        self.Close()




class Misc(wx.Dialog):

    def __init__(self, parent, id, title):
        wx.Dialog.__init__(self, parent, id, title,
                        wx.DefaultPosition, wx.Size(350,510))

        vbox = wx.BoxSizer(wx.VERTICAL)
        lev_text = wx.StaticText(self, -1, 'Minimum level of Countrous [in %]', style = wx.ALIGN_CENTER | wx.ALL )
        self.min_lev = wx.Slider(self, -1, 5, 1, 10., wx.DefaultPosition, (250,-1),
                                                                         wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS )
        vbox.Add(lev_text, 1, wx.ALIGN_CENTER | wx.EXPAND)
        vbox.Add(self.min_lev, 1, wx.ALIGN_CENTRE)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        xmin_t = wx.StaticText(self, -1,        'Minimum X', style = wx.ALIGN_LEFT)
        self.xmin_v = wx.TextCtrl(self, -1, size=(25,-1), value= str(0), style=wx.TE_PROCESS_ENTER)
        hbox1.Add(xmin_t, 1, wx.ALIGN_CENTRE)
        hbox1.Add(self.xmin_v, 1, wx.ALIGN_CENTER)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        xmax_t = wx.StaticText(self, -1,        'Maximum X', style = wx.ALIGN_LEFT)
        self.xmax_v = wx.TextCtrl(self, -1, size=(25,-1), value= str(0), style=wx.TE_PROCESS_ENTER)
        hbox2.Add(xmax_t, 1, wx.ALIGN_CENTER)
        hbox2.Add(self.xmax_v, 1, wx.ALIGN_CENTER)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        ymin_t = wx.StaticText(self, -1,        'Minimum Y', style = wx.ALIGN_LEFT)
        self.ymin_v = wx.TextCtrl(self, -1, size=(25,-1), value= str(0), style=wx.TE_PROCESS_ENTER)
        hbox3.Add(ymin_t, 1, wx.ALIGN_CENTER)
        hbox3.Add(self.ymin_v, 1, wx.ALIGN_CENTER)

        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        ymax_t = wx.StaticText(self, -1,        'Maximum Y', style = wx.ALIGN_LEFT)
        self.ymax_v = wx.TextCtrl(self, -1, size=(25,-1), value= str(0), style=wx.TE_PROCESS_ENTER)
        hbox4.Add(ymax_t, 1, wx.ALIGN_CENTER)
        hbox4.Add(self.ymax_v, 1, wx.ALIGN_CENTER)

        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        btn1 = wx.Button(self, 253, 'Adjust')
        btn2 = wx.Button(self, 254, 'Close')
        hbox5.Add(btn1, 1, wx.ALIGN_CENTER)
        hbox5.Add(btn2, 1, wx.ALIGN_CENTER)

        vbox.Add(hbox1, 1, wx.ALIGN_CENTER)
        vbox.Add(hbox2, 1, wx.ALIGN_CENTER)
        vbox.Add(hbox3, 1, wx.ALIGN_CENTER)
        vbox.Add(hbox4, 1, wx.ALIGN_CENTER)
        vbox.Add(hbox5, 1, wx.ALIGN_CENTER)

        self.SetSizer(vbox)

        self.Bind(wx.EVT_BUTTON, self.GetVals, id=253)
        self.Bind(wx.EVT_BUTTON, self.OnQuit, id=254)


    def GetVals(self, event):

        print("\n"+"*"*10)
        print("Image setting changed. \n")
        print("Minimum level = "+str(0.1*self.min_lev.GetValue())+'Percent \n')
        print("Xmin = "+str(self.xmin_v.GetValue())+" pixel \n")
        print("Xmax = "+str(self.xmax_v.GetValue())+" pixel \n")
        print("Ymin = "+str(self.ymin_v.GetValue())+" pixel \n")
        print("Ymax = "+str(self.ymax_v.GetValue())+" pixel \n")

        self.update_xy = np.array( [self.min_lev.GetValue()*1e-3,
                                                        self.xmin_v.GetValue(),
                                                        self.xmax_v.GetValue(),
                                                        self.ymin_v.GetValue(),
                                                        self.ymax_v.GetValue()] , dtype=float)

        self.min_lev.Value = self.min_lev.GetValue()


    def OnQuit(self, event):
        self.Close()



class ImgBoxCtrl(wx.Panel):

    def __init__(self, parent, ID, label, initval):
        wx.Panel.__init__(self, parent, ID)

        self.value = initval

        box = wx.StaticBox(self, -1, label)
        sizer= wx.StaticBoxSizer(box, wx.HORIZONTAL)

        self.radio_img1 = wx.RadioButton(self, -1,
                        label="Image 1",style=wx.RB_GROUP)
        self.radio_img2 = wx.RadioButton(self, -1,
                        label="Image 2")

        sizer.Add(self.radio_img1, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT | wx.LEFT | wx.EXPAND, 1 )
        sizer.Add(self.radio_img2, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT | wx.LEFT | wx.EXPAND, 1)

        self.SetSizer(sizer)
        self.Centre()

    def img_value(self):
        return self.radio_img1.GetValue()


class SetMask(wx.Panel):

    def __init__(self, parent, ID, label, init_val):
        wx.Panel.__init__(self, parent, ID)

        self.init_val = init_val

        box = wx.StaticBox(self, -1, label)
        sizer= wx.StaticBoxSizer(box, wx.VERTICAL)

        self.xcenter= wx.TextCtrl(self, -1,
                        size=(25,-1),
                        value=str(self.init_val[0]),
                        style=wx.TE_PROCESS_ENTER)
        self.ycenter= wx.TextCtrl(self, -1,
                        size=(25,-1),
                        value=str(self.init_val[1]),
                        style=wx.TE_PROCESS_ENTER)
        self.major= wx.TextCtrl(self, -1,
                        size=(25,-1),
                        value=str(self.init_val[2]),
                        style=wx.TE_PROCESS_ENTER)
        self.minor= wx.TextCtrl(self, -1,
                        size=(25,-1),
                        value=str(self.init_val[3]),
                        style=wx.TE_PROCESS_ENTER)
        self.tilt= wx.TextCtrl(self, -1,
                        size=(25,-1),
                        value=str(self.init_val[4]),
                        style=wx.TE_PROCESS_ENTER)
        self.xtext = wx.StaticText(self, -1, 'X [pix]', style=wx.ALIGN_LEFT)
        self.ytext = wx.StaticText(self, -1, 'Y [pix]', style=wx.ALIGN_LEFT)
        self.major_text= wx.StaticText(self, -1, 'a [pix]', style=wx.ALIGN_LEFT)
        self.minor_text= wx.StaticText(self, -1, 'b [pix]', style=wx.ALIGN_LEFT)
        self.tilt_text= wx.StaticText(self, -1, u'\u0398'+' [deg]', style=wx.ALIGN_LEFT)

        font = wx.Font(10, wx.ROMAN, wx.NORMAL, wx.NORMAL)
        self.xtext.SetFont(font)
        self.ytext.SetFont(font)
        self.major_text.SetFont(font)
        self.minor_text.SetFont(font)
        self.tilt_text.SetFont(font)

        small_box1 = wx.BoxSizer(wx.HORIZONTAL)
        small_box1.Add(self.xtext, 1, wx.ALIGN_CENTER, 10)
        small_box1.Add(self.xcenter, 1, wx.ALIGN_RIGHT, 10)

        small_box2 = wx.BoxSizer(wx.HORIZONTAL)
        small_box2.Add(self.ytext, 1, wx.ALIGN_CENTRE, 10)
        small_box2.Add(self.ycenter, 1, wx.ALIGN_RIGHT, 10)

        small_box3 = wx.BoxSizer(wx.HORIZONTAL)
        small_box3.Add(self.major_text, 1, wx.ALIGN_CENTRE, 10)
        small_box3.Add(self.major, 1, wx.ALIGN_RIGHT, 10)

        small_box4 = wx.BoxSizer(wx.HORIZONTAL)
        small_box4.Add(self.minor_text, 1, wx.ALIGN_CENTRE, 10)
        small_box4.Add(self.minor, 1, wx.ALIGN_RIGHT, 10)

        small_box5 = wx.BoxSizer(wx.HORIZONTAL)
        small_box5.Add(self.tilt_text, 1, wx.ALIGN_CENTRE, 10)
        small_box5.Add(self.tilt, 1, wx.ALIGN_RIGHT | wx.EXPAND, 10)

        sizer.Add(small_box1, 1, wx.ALL,10)
        sizer.Add(small_box2, 1, wx.ALL,10)
        sizer.Add(small_box3, 1, wx.ALL,10)
        sizer.Add(small_box4, 1, wx.ALL,10)
        sizer.Add(small_box5, 1, wx.ALL,10)

        self.SetSizer(sizer)
        self.Centre()
        sizer.Fit(self)

    def get_center(self):
        return np.array([self.xcenter.GetValue(), self.ycenter.GetValue()], dtype=float)
    def ellipse_get(self):
        return np.array( [self.major.GetValue(), self.minor.GetValue(), self.tile.GetValue() ], dtype=float )


class SizeSetting(wx.Panel):

    def __init__(self, parent, ID, label, init_val):
        wx.Panel.__init__(self, parent, ID)

        self.init_val = init_val

        box = wx.StaticBox(self, -1, label)
        sizer= wx.StaticBoxSizer(box, wx.VERTICAL)

        self.xmin= wx.TextCtrl(self, -1,
                        size=(25,-1),
                        value=str(self.init_val[0]),
                        style=wx.TE_PROCESS_ENTER)
        self.xmax= wx.TextCtrl(self, -1,
                        size=(25,-1),
                        value=str(self.init_val[1]),
                        style=wx.TE_PROCESS_ENTER)
        self.ymin= wx.TextCtrl(self, -1,
                        size=(25,-1),
                        value=str(self.init_val[2]),
                        style=wx.TE_PROCESS_ENTER)
        self.ymax= wx.TextCtrl(self, -1,
                        size=(25,-1),
                        value=str(self.init_val[3]),
                        style=wx.TE_PROCESS_ENTER)

        self.xmin_t = wx.StaticText(self, -1, 'min X', style=wx.ALIGN_LEFT)
        self.xmax_t = wx.StaticText(self, -1, 'max X', style=wx.ALIGN_LEFT)
        self.ymin_t = wx.StaticText(self, -1, 'min Y', style=wx.ALIGN_LEFT)
        self.ymax_t = wx.StaticText(self, -1, 'max Y', style=wx.ALIGN_LEFT)

        font = wx.Font(10, wx.ROMAN, wx.NORMAL, wx.NORMAL)
        self.xmin_t.SetFont(font)
        self.xmax_t.SetFont(font)
        self.ymin_t.SetFont(font)
        self.ymax_t.SetFont(font)

        small_box1 = wx.BoxSizer(wx.HORIZONTAL)
        small_box1.Add(self.xmin_t, 1, wx.ALIGN_CENTER, 10)
        small_box1.Add(self.xmin, 1, wx.ALIGN_RIGHT, 10)

        small_box2 = wx.BoxSizer(wx.HORIZONTAL)
        small_box2.Add(self.xmax_t, 1, wx.ALIGN_CENTRE, 10)
        small_box2.Add(self.xmax, 1, wx.ALIGN_RIGHT, 10)

        small_box3 = wx.BoxSizer(wx.HORIZONTAL)
        small_box3.Add(self.ymin_t, 1, wx.ALIGN_CENTRE, 10)
        small_box3.Add(self.ymin, 1, wx.ALIGN_RIGHT, 10)

        small_box4 = wx.BoxSizer(wx.HORIZONTAL)
        small_box4.Add(self.ymax_t, 1, wx.ALIGN_CENTRE, 10)
        small_box4.Add(self.ymax, 1, wx.ALIGN_RIGHT, 10)

        gs1 = wx.GridSizer(2,2,6,6)
        gs1.AddMany([ (small_box1,1, wx.ALIGN_CENTER),
                                                (small_box2,1, wx.ALIGN_CENTER),
                                                (small_box3,1, wx.ALIGN_CENTER),
                                                (small_box4,1, wx.ALIGN_CENTER) ])

        sizer.Add(gs1, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL,10)

        self.SetSizer(sizer)
        self.Centre()

    def get_center(self):
        return np.array([self.xcenter.GetValue(), self.ycenter.GetValue()], dtype=float)
    def ellipse_get(self):
        return np.array( [self.major.GetValue(), self.minor.GetValue(), self.tile.GetValue() ], dtype=float )



class CorrFrame(wx.Frame):


    def __init__(self, parent, id, title):

        wx.Frame.__init__(self, parent, id, title, wx.DefaultPosition, wx.Size(500,400))

        self.panel = wx.Panel(self,-1)

        self.img1 = TopFrame.masked_img1
        self.img2 = TopFrame.masked_img2

        self.result = self.CorrFunc()
        self.ImShow()

        sizer_v = wx.BoxSizer(wx.VERTICAL)
        sizer_v.Add( self.canvas , proportion=1, flag= wx.TOP | wx.CENTER | wx.GROW, border = 0)

        sizer_h = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h.Add( self.toolbar, proportion=0, flag=wx.EXPAND | wx.ALIGN_CENTER, border=0)
        sizer_h.Add( wx.Button(self.panel, 301, 'Done'), proportion=1, flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER | wx.LEFT | wx.EXPAND, border=350)

        sizer_v.Add( sizer_h, proportion=0, flag=wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=10)

        self.panel.SetSizer(sizer_v)
        sizer_v.Fit(self)

        self.Bind(wx.EVT_BUTTON, self.OnExit, id=301)

    def OnExit(self, event):
        print("\nYou can additionally check spectral index distribution.")
        print("To do that, please click 'SPIX map' menu.\n" )
        self.Close()

    def CorrFunc(self):

        "Practical computation part"

        self.n  = np.int( len(self.img1[0,:]) )
        self.n2 = np.int( len(self.img1[:,0]) )
        self.bb = np.int(1./5*2*np.int(self.n2)-1)
        self.ee = np.int(1./5*2*np.int(self.n)-1)
#self.cor_arr = np.zeros( [1./5*2*np.int(self.n2)-1, 1./5*2*np.int(self.n)-1] )
        self.cor_arr = np.zeros( [self.bb, self.ee] )

        count_tmp = 0

        print("\n...")
        print("It may take some time ...")
        print("...\n")

        for x in np.arange( np.int(4./5* self.n) ,  np.int(6./5* self.n-1) , 1 ):
            for y in np.arange( np.int(4./5* self.n2) ,  np.int(6./5* self.n2-1) , 1):

                if x< self.n:
                    if y< self.n2:
                        self.temp1= self.img1[ 0:y, 0:x ]
                        self.temp2= self.img2[ (self.n2-1)-y:(self.n2-1), (self.n-1)-x:(self.n-1) ]
                    else:
                        self.temp1= self.img1[ y-(self.n2-1):(self.n2-1) , 0:x ]
                        self.temp2= self.img2[ 0:2*(self.n2-1)-y,(self.n-1)-x:(self.n-1) ]
                else:
                    if y< self.n2:
                        self.temp1= self.img1[ 0:y, x-(self.n-1):self.n-1 ]
                        self.temp2= self.img2[ (self.n2-1)-y:(self.n2-1), 0:2*(self.n-1)-x ]
                    else:
                        self.temp1= self.img1[ y-(self.n2-1):(self.n2-1),x-(self.n-1):self.n-1 ]
                        self.temp2= self.img2[ 0:2*(self.n2-1)-y,0:2*(self.n-1)-x ]

                self.up = np.sum( ( self.temp1 - np.mean(self.img1))*( self.temp2 - np.mean(self.img2) ) )
                self.down = np.sqrt( np.sum(  (self.temp1 - np.mean(self.img1))**2 ) * np.sum(  (self.temp2 - np.mean(self.img2))**2 ) )

                y_ind = np.int(y- (4./5*self.n2 ))
                x_ind = np.int(x- (4./5*self.n  ))

                if np.float(self.down)==0:
                    self.cor_arr[y_ind,x_ind] = np.nan
                else:
                    self.cor_arr[y_ind,x_ind] = np.copy( np.float(self.up)/np.float(self.down) )


        self.xr= np.arange( -0.2*self.n+1 , 0.2*self.n,  1)
        self.yr= np.arange( -0.2*self.n2+1, 0.2*self.n2, 1)

        self.ind = ( self.cor_arr == np.copy(self.cor_arr.max()) ).nonzero()

        self.xr_ind = (self.ind)[1]
        self.yr_ind = (self.ind)[0]

        self.max_x = self.xr[ self.xr_ind ]
        self.max_y = self.yr[ self.yr_ind ]

        TopFrame.shift_x = np.copy(self.max_x)
        TopFrame.shift_y = np.copy(self.max_y)
        TopFrame.cor_val = np.copy(self.cor_arr[self.ind])
        TopFrame.shift_mas = np.copy(np.sqrt( (self.max_x*TopFrame.dxmas)**2 + (self.max_y*TopFrame.dymas)**2 ))

        print('\n', '*'*10)
        print('\nCalculation finished ! \n')
        print('VIMAP suggests you to shift your high frequency image by;\n')
        print('in RA  : ', str(self.max_x)+' Pixel')
        print('in DEC : ', str(self.max_y)+' Pixel')
        print('in RA  : ', str(self.max_x*TopFrame.dxmas) + ' mas')
        print('in DEC  : ', str(self.max_y*TopFrame.dymas) + ' mas')
        print('Max cross-corr value at there : '+str(self.cor_arr[self.ind]))
        print('Shift in angle : '+str(   np.sqrt( (self.max_x*TopFrame.dxmas)**2 + (self.max_y*TopFrame.dymas)**2 ) )+' MilliArc Second')

        print('\nPlease check these values through interactive plot window.')
        print('you should find any other peaks in the map\nif the peak does not geometrically make sense.')
        print('\n', '*'*10, '\n')


        return np.copy( self.cor_arr )

    def ImShow(self):

        self.dpi = 150
        self.fig = Figure((5.0,4.0), dpi=self.dpi)
        self.canvas = FigCanvas(self.panel, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        self.axes = self.fig.add_subplot(111)
        self.ml = MultipleLocator(5)

        Art.setp(self.axes.get_xticklabels(), fontsize=9)
        Art.setp(self.axes.get_yticklabels(), fontsize=9)

        self.axes.set_xlabel('shift in RA [Pixel]')
        self.axes.set_ylabel('shift in DEC [Pixel]')
        self.axes.set_title('Result of 2D Correlation')

        self.axes.xaxis.set_minor_locator(self.ml)
        self.axes.yaxis.set_minor_locator(self.ml)

        self.ext_range = [ -(0.2*self.n -1), 0.2*self.n-1, -(0.2*self.n2 -1), 0.2*self.n2 -1 ]

        self.show_img = self.axes.imshow( self.result, origin='lower', extent=self.ext_range, cmap=cm.gist_ncar, picker=True)
        self.show_cnt = self.axes.contour( self.result, np.arange(0.7,1,0.03), colors='k', origin='lower', extent= self.ext_range )
        cbar= self.fig.colorbar( self.show_img, ticks=[-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0] )
        self.axes.tick_params(labelsize=15)

        self.axes.axhline(y=0, xmin=-9999, xmax=9999, linewidth=2, color='w', linestyle='--')
        self.axes.axvline(x=0, ymin=-9999, ymax=9999, linewidth=2, color='w', linestyle='--')

        self.canvas.draw()



class SpixFrame(wx.Frame):


    def __init__(self, parent, id, title):

        wx.Frame.__init__(self, parent, id, title, wx.DefaultPosition, wx.Size(700,600))

        panel = wx.Panel(self, -1, style=wx.SUNKEN_BORDER)

        self.img1 = np.copy( TopFrame.raw_img1 )
        self.img2 = np.copy( TopFrame.raw_img2 )

        self.shift_x = np.copy( TopFrame.shift_x )
        self.shift_y = np.copy( TopFrame.shift_y )

        self.ShowPanel()
        self.CalcSpix()

        self.dpi = 150
        self.fig = Figure((5.0,4.0), dpi=self.dpi)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        self.new_plot=0

        self.ShowResult()

        bigsizer = wx.BoxSizer(wx.HORIZONTAL)

        sizer_small = wx.BoxSizer(wx.VERTICAL)
        sizer_small.Add(self.canvas, 1, wx.ALIGN_CENTRE | wx.RIGHT | wx.EXPAND, border=0)
        sizer_small.Add(self.toolbar, proportion=0, flag=wx.EXPAND | wx.ALIGN_CENTER, border=0)

        bigsizer.Add(sizer_small, 1, wx.ALIGN_CENTER | wx.RIGHT | wx.EXPAND, border=0)
        bigsizer.Add(self.sizer_v, 0, wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, 1)

        self.SetSizer(bigsizer)
        bigsizer.Fit(self)
        self.Centre()


    def ShowPanel(self):


        if TopFrame.freq1 != 0:
            if TopFrame.freq2 != 0:
                freq_arr = [ np.copy( TopFrame.freq1 ), np.copy( TopFrame.freq2)  ]
        else:
            freq_arr = np.array( [10., 100.], dtype='float' )
            print('Program fails to read frequency information from FITS files.')
            print('Please put the values manually.')

        self.freq1_t = wx.StaticText(self, -1, 'Freq 1 [GHz]', style=wx.ALIGN_LEFT)
        self.freq1_v = wx.TextCtrl(self, -1,
                        size=(50,-1),
                        value=str( np.round( freq_arr[0]/1e+9, decimals=3) ),
                        style=wx.TE_PROCESS_ENTER)
        self.freq2_t = wx.StaticText(self, -1, 'Freq 2 [GHz]', style=wx.ALIGN_LEFT)
        self.freq2_v = wx.TextCtrl(self, -1,
                        size=(50,-1),
                        value=str( np.round( freq_arr[1]/1e+9, decimals=3) ),
                        style=wx.TE_PROCESS_ENTER)

        box1 = wx.StaticBox(self, -1, '**Frequency**')

        self.cb_freq = wx.CheckBox(self, -1,
                        "Allow different freq",
                        style=wx.ALIGN_RIGHT)

        sizer_v1 = wx.StaticBoxSizer(box1, wx.VERTICAL)

        sizer_h11 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h11.Add(self.freq1_t, 1, wx.ALIGN_CENTER, 10)
        sizer_h11.Add(self.freq1_v, 1, wx.ALIGN_CENTER | wx.ALIGN_RIGHT, 10)

        sizer_h12 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h12.Add(self.freq2_t, 1, wx.ALIGN_CENTER, 10)
        sizer_h12.Add(self.freq2_v, 1, wx.ALIGN_CENTER | wx.ALIGN_RIGHT, 10)

        sizer_v1.Add(self.cb_freq, 1, wx.ALIGN_CENTER | wx.ALL, 5)
        sizer_v1.Add(sizer_h11, 1, wx.ALIGN_CENTER , 3)
        sizer_v1.Add(sizer_h12, 1, wx.ALIGN_CENTER , 3)



        rms1_t      = wx.StaticText(self, -1, 'RMS 1 [mJy/bm]', style=wx.ALIGN_LEFT)
        self.rms1_v = wx.TextCtrl(self, -1,
                        size=(50,-1),
                        value=str(10),
                        style=wx.TE_PROCESS_ENTER)
        rms2_t      = wx.StaticText(self, -1, 'RMS 2 [mJy/bm]', style=wx.ALIGN_LEFT)
        self.rms2_v = wx.TextCtrl(self, -1,
                        size=(50,-1),
                        value=str(10),
                        style=wx.TE_PROCESS_ENTER)

        cal_err1t = wx.StaticText(self, -1, 'Calib. Err 1 [%]', style=wx.ALIGN_LEFT)
        self.cal_err1v = wx.TextCtrl(self, -1,
                        size=(50,-1),
                        value=str(5.0),
                        style=wx.TE_PROCESS_ENTER)
        cal_err2t = wx.StaticText(self, -1, 'Calib. Err 2 [%]', style=wx.ALIGN_LEFT)
        self.cal_err2v = wx.TextCtrl(self, -1,
                        size=(50,-1),
                        value=str(5.0),
                        style=wx.TE_PROCESS_ENTER)
        box2 = wx.StaticBox(self, -1, 'Error estimation')


        sizer_v2 = wx.StaticBoxSizer(box2, wx.VERTICAL)

        sizer_h21 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h21.Add(rms1_t, 1, wx.ALIGN_CENTER)
        sizer_h21.Add(self.rms1_v, 1, wx.ALIGN_RIGHT | wx.ALIGN_CENTER)

        sizer_h22 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h22.Add(rms2_t, 1, wx.ALIGN_CENTER)
        sizer_h22.Add(self.rms2_v, 1, wx.ALIGN_RIGHT | wx.ALIGN_CENTER)

        sizer_h23 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h23.Add(cal_err1t, 1, wx.ALIGN_CENTER)
        sizer_h23.Add(self.cal_err1v, 1, wx.ALIGN_RIGHT | wx.ALIGN_CENTER)

        sizer_h24 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h24.Add(cal_err2t, 1, wx.ALIGN_CENTER)
        sizer_h24.Add(self.cal_err2v, 1, wx.ALIGN_RIGHT | wx.ALIGN_CENTER)

        sizer_v2.Add(sizer_h21, 1, wx.ALIGN_CENTER)
        sizer_v2.Add(sizer_h22, 1, wx.ALIGN_CENTER)
        sizer_v2.Add(sizer_h23, 1, wx.ALIGN_CENTER)
        sizer_v2.Add(sizer_h24, 1, wx.ALIGN_CENTER)



        sizer_v3 = wx.BoxSizer(wx.VERTICAL)
        sld1_text = wx.StaticText(self, -1, 'Cutoff by Lowest Flux [in 0.01%]', style= wx.ALIGN_CENTER)
        self.sld1 = wx.Slider(self, -1,
                        2,1,150, wx.DefaultPosition, (250,-1),
                        wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS, name='Cutoff [in flux]')
        self.sld1.SetTickFreq(1)
        sld2_text = wx.StaticText(self, -1, 'Cutoff by SPIX Error [in 0.1 unit]', style= wx.ALIGN_CENTER )
        self.sld2 = wx.Slider(self, -1,
                        5.0,0.0,10.0, wx.DefaultPosition, (250,-1),
                        wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS, name='Cutoff [in spix.err]')
        self.sld2.SetTickFreq(1)

        sizer_v3.Add(sld1_text, 0, wx.ALIGN_CENTER,1)
        sizer_v3.Add(self.sld1, 0, wx.ALIGN_CENTER,1)
        sizer_v3.Add(sld2_text, 0, wx.ALIGN_CENTER,1)
        sizer_v3.Add(self.sld2, 0, wx.ALIGN_CENTER,1)


        sizer_v4 = wx.BoxSizer(wx.VERTICAL)

        self.cb_shift = wx.CheckBox(self, -1,
                        "Allow manual shift",
                        style=wx.ALIGN_RIGHT)

        shift1_t = wx.StaticText(self, -1, 'X direction', style=wx.ALIGN_LEFT)
        self.shift1_v= wx.TextCtrl(self, -1,
                        size=(50,-1),
                        value=str( self.shift_x[0] ),
                        style=wx.TE_PROCESS_ENTER)
        shift2_t = wx.StaticText(self, -1, 'Y direction', style=wx.ALIGN_LEFT)
        self.shift2_v= wx.TextCtrl(self, -1,
                        size=(50,-1),
                        value=str( self.shift_y[0] ),
                        style=wx.TE_PROCESS_ENTER)

        box4 = wx.StaticBox(self, -1, '**Manual Shift**')

        sizer_v4 = wx.StaticBoxSizer(box4, wx.VERTICAL)

        sizer_h41 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h41.Add(shift1_t, 1, wx.ALIGN_CENTER)
        sizer_h41.Add(self.shift1_v, 1, wx.ALIGN_RIGHT | wx.ALIGN_CENTER)

        sizer_h42 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h42.Add(shift2_t, 1, wx.ALIGN_CENTER)
        sizer_h42.Add(self.shift2_v, 1, wx.ALIGN_RIGHT | wx.ALIGN_CENTER)


        sizer_v4.Add(self.cb_shift, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        sizer_v4.Add(sizer_h41, 0, wx.ALIGN_CENTER,1)
        sizer_v4.Add(sizer_h42, 0, wx.ALIGN_CENTER,1)




        self.sizer_v = wx.BoxSizer(wx.VERTICAL)
        self.sizer_v.Add(sizer_v1, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, 10)
        self.sizer_v.Add(sizer_v4, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, 10)
        self.sizer_v.Add(sizer_v3, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, 10)
        self.sizer_v.Add(sizer_v2, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, 10)



        gs = wx.GridSizer(4,1,1,0)
        gs.AddMany( [ (wx.Button(self, 401,'PLOT SPIX'), 1, wx.RIGHT | wx.LEFT | wx.ALIGN_CENTER | wx.EXPAND, 5 ),
                                          (wx.Button(self, 402,'PLOT ERROR'),1, wx.RIGHT | wx.LEFT | wx.ALIGN_CENTER | wx.EXPAND, 5 ),
                                          (wx.Button(self, 403,'SAVE PARAMS'),1, wx.RIGHT | wx.LEFT | wx.ALIGN_CENTER | wx.EXPAND, 5 ),
                                          (wx.Button(self, 404,'EXIT'),      1, wx.RIGHT | wx.LEFT | wx.ALIGN_CENTER | wx.EXPAND, 5 )] )

        self.Bind(wx.EVT_BUTTON, self.OnPLOT, id=401)
        self.Bind(wx.EVT_BUTTON, self.OnPERR, id=402)
        self.Bind(wx.EVT_BUTTON, self.OnSAVE, id=403)
        self.Bind(wx.EVT_BUTTON, self.OnEXIT, id=404)

        self.sizer_v.Add(gs, 0, wx.EXPAND | wx.ALIGN_BOTTOM | wx.ALIGN_CENTER | wx.ALL, border=10)

    def CalcSpix(self):


        if self.cb_shift.GetValue():
            try:
                self.shift_x = np.float( self.shift1_v.GetValue() )
                self.shift_y = np.float( self.shift2_v.GetValue() )
            except:
                print("Please check the manual shift setup.")

        else:
            self.shift_x = np.float(TopFrame.shift_x)
            self.shift_y = np.float(TopFrame.shift_y)

        if self.cb_freq.GetValue():
            try:
                self.freq1 = np.float( self.freq1_v.GetValue() )
                self.freq2 = np.float( self.freq2_v.GetValue() )
            except:
                print("Please check the manual frequency setup.")
        else:
            self.freq1 = TopFrame.freq1
            self.freq2 = TopFrame.freq2

        self.shift_x = np.int(self.shift_x)
        self.shift_y = np.int(self.shift_y)

        if self.shift_x >0 :
            if self.shift_y > 0 :
                self.spix2 = np.copy( self.img2[ : -np.abs(self.shift_y) ,  : -np.abs(self.shift_x)] )
                self.spix1 = np.copy( self.img1[ np.abs(self.shift_y) :  , np.abs(self.shift_x) : ] )
            elif self.shift_y <0 :
                self.spix2 = np.copy( self.img2[ np.abs(self.shift_y) :  ,  : -np.abs(self.shift_x)] )
                self.spix1 = np.copy( self.img1[ : -np.abs(self.shift_y) , np.abs(self.shift_x) : ] )
            elif self.shift_y==0 :
                self.spix2 = np.copy( self.img2[ :  ,  : -np.abs(self.shift_x)] )
                self.spix1 = np.copy( self.img1[ :  , np.abs(self.shift_x) : ] )
        elif self.shift_x <0 :
            if self.shift_y > 0 :
                self.spix2 = np.copy( self.img2[ : -np.abs(self.shift_y) ,  np.abs(self.shift_x) : ] )
                self.spix1 = np.copy( self.img1[ np.abs(self.shift_y) :  ,  : -np.abs(self.shift_x)] )
            elif self.shift_y <0 :
                self.spix2 = np.copy( self.img2[ np.abs(self.shift_y) : ,  np.abs(self.shift_x) : ] )
                self.spix1 = np.copy( self.img1[ : -np.abs(self.shift_y) ,  : -np.abs(self.shift_x)] )
            elif self.shift_y==0 :
                self.spix2 = np.copy( self.img2[ : ,  np.abs(self.shift_x) : ] )
                self.spix1 = np.copy( self.img1[ : ,  : -np.abs(self.shift_x)] )
        elif self.shift_x==0 :
            if self.shift_y > 0 :
                self.spix2 = np.copy( self.img2[ : -np.abs(self.shift_y) ,  : ] )
                self.spix1 = np.copy( self.img1[ np.abs(self.shift_y) :  ,  : ] )
            elif self.shift_y <0 :
                self.spix2 = np.copy( self.img2[ np.abs(self.shift_y) : ,  : ] )
                self.spix1 = np.copy( self.img1[ : -np.abs(self.shift_y) ,  : ] )
            elif self.shift_y==0 :
                self.spix2 = np.copy( self.img2[ : ,  : ] )
                self.spix1 = np.copy( self.img1[ : ,  : ] )

        self.freq1= np.float( self.freq1_v.GetValue() )
        self.freq2= np.float( self.freq2_v.GetValue() )

        self.spix = np.log10( self.spix2/self.spix1 )/np.log10( self.freq2/self.freq1 )
        self.spix_backup = np.copy( self.spix )

        self.CalcError()

        self.spix_for_replot = np.copy( self.spix )
        self.spix_for_replot[ self.spix1 >= 1e-3*self.spix_cut*self.spix1.max() ] = np.median(self.spix_for_replot)
        self.spix_for_replot[ self.err_tot <= self.spix_err_limit ] = np.median(self.spix_for_replot)

        self.spix[ self.spix1 < 1e-3*self.spix_cut*self.spix1.max() ] = np.nan
        self.spix[ self.err_tot > self.spix_err_limit ] = np.nan

    def CalcError(self):

        self.rms1= 1e-3*np.float(self.rms1_v.GetValue())
        self.rms2= 1e-3*np.float(self.rms2_v.GetValue())
        self.cal_err1= 1e-2*np.float( self.cal_err1v.GetValue() )
        self.cal_err2= 1e-2*np.float( self.cal_err2v.GetValue() )
        self.spix_cut = np.float( self.sld1.GetValue() )
        self.spix_err_limit = 0.1*np.float( self.sld2.GetValue() )

        self.noise1 = self.cal_err1*self.spix1 + self.rms1
        self.noise2 = self.cal_err2*self.spix2 + self.rms2

        self.err1 = self.noise1/self.spix1 * 1/(np.log10(self.freq2/self.freq1)) * 1/np.log(10)
        self.err2 = self.noise2/self.spix2 * 1/(np.log10(self.freq2/self.freq1)) * 1/np.log(10)

        self.err_tot = np.sqrt( self.err1**2 + self.err2**2 )


    def ShowResult(self):

        self.PlotSetting()
        self.plot_kwargs = { 'origin' : 'lower',
                'extent':[0.5*len(self.spix[0,:])*np.abs(TopFrame.dxmas),-0.5*len(self.spix[0,:])*np.abs(TopFrame.dymas),
                                        -0.5*len(self.spix[:,0])*np.abs(TopFrame.dymas), 0.5*len(self.spix[:,0])*np.abs(TopFrame.dymas)]}

        self.levels = 2. ** np.arange( np.log10(self.spix_cut*1e-3*self.spix1.max())/np.log10(2), np.log10(self.spix1.max())/np.log10(2), 0.5)
        np.append( -1*2.**( np.log10(self.spix_cut*1e-3*self.spix1.max())/np.log10(2)  ) , self.levels)

        self.show_img = self.axes.imshow( self.spix , interpolation='none',cmap=cm.jet, vmax=1.5, vmin=-2.5,**self.plot_kwargs)
        self.axes.tick_params(labelsize=15)

        self.show_cnt = self.axes.contour( self.spix1 , self.levels, colors='k', **self.plot_kwargs)

        if self.new_plot==0:
            self.cbar= self.fig.colorbar( self.show_img, ticks= [-2.5,-2.0,-1.5,-1,-0.5,0,0.5,1,1.5] )
        else:
            #self.cbar.set_clim( vmax= 2.0, vmin=-2.0 )
            self.cbar.draw_all()

        plt.minorticks_on()
        self.canvas.draw()

    def ShowErrTot(self):

        self.PlotSetting_forErr()
        self.plot_kwargs = { 'origin' : 'lower',
                'extent':[0.5*len(self.spix[0,:])*np.abs(TopFrame.dxmas),-0.5*len(self.spix[0,:])*np.abs(TopFrame.dymas),
                                        -0.5*len(self.spix[:,0])*np.abs(TopFrame.dymas), 0.5*len(self.spix[:,0])*np.abs(TopFrame.dymas)],
                'vmax':1, 'vmin':0}

        self.levels = 2. ** np.arange( np.log10(self.spix_cut*1e-3*self.spix1.max())/np.log10(2), np.log10(self.spix1.max())/np.log10(2), 0.5)
        np.append( -1*2.**( np.log10(self.spix_cut*1e-3*self.spix1.max())/np.log10(2)  ) , self.levels)

        self.show_img = self.axes.imshow( self.err_tot, cmap=cm.jet, **self.plot_kwargs)
        self.axes.tick_params(labelsize=15)
        self.show_cnt = self.axes.contour( self.spix1 , self.levels, colors='k', **self.plot_kwargs)

        if self.new_plot==0:
            self.cbar= self.fig.colorbar( self.show_img, ticks= [0,0.2,0.4,0.6,0.8,1] )
        else:
            #self.cbar.set_clim( vmax=1., vmin=0. )
            self.cbar.draw_all()

        self.canvas.draw()

    def sizeHandler(self, *args, **kwargs):
        self.canvas.SetSize(self.GetSize())

    def PlotSetting(self):

        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('white')
        self.ml = MultipleLocator(5)

        Art.setp(self.axes.get_xticklabels(), fontsize=13)
        Art.setp(self.axes.get_yticklabels(), fontsize=13)

        self.axes.set_xlabel('Relative RA [mas]')
        self.axes.set_ylabel('Relative DEC [mas]')
        self.axes.set_title('Spectral Index Distribution')

        self.axes.xaxis.set_minor_locator(self.ml)
        self.axes.yaxis.set_minor_locator(self.ml)

    def PlotSetting_forErr(self):

        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('white')
        self.ml = MultipleLocator(5)

        Art.setp(self.axes.get_xticklabels(), fontsize=13)
        Art.setp(self.axes.get_yticklabels(), fontsize=13)

        self.axes.set_xlabel('Relative RA [mas]')
        self.axes.set_ylabel('Relative DEC [mas]')
        self.axes.set_title('Spectral Index Error Distribution')

        self.axes.xaxis.set_minor_locator(self.ml)
        self.axes.yaxis.set_minor_locator(self.ml)

    def OnPLOT(self, event):
        self.new_plot =1
        self.CalcSpix()
        self.axes.clear()
        self.ShowResult()

    def OnPERR(self, event):
        self.new_plot =1
        self.CalcSpix()
        self.axes.clear()
        self.ShowErrTot()

    def OnSAVE(self, event):
        #MAS2RAD = np.pi/(6.48*10**8)

        self.out_list = glob('VIMAP_output*.dat')
        self.output_v = 1+len(self.out_list)

        f = open('VIMAP_output'+str(self.output_v)+'.dat','w')

        f.write('#'*30)
        f.write('\n# Output from VIMAP v1.2\n')
        f.write('#'*30)
        f.write('\n\n\n\n')

        f.write('# Path to the FITS\n')
        f.write('Low freq FITS file   : '+ TopFrame.filepath[0]+'\n')
        f.write('High freq FITS file  : '+ TopFrame.filepath[1]+'\n')
        f.write('\n')

        f.write('# Resize Parameters\n')
        f.write('x [min] : '+str(TopFrame.re_xmin)+'\n')
        f.write('x [max] : '+str(TopFrame.re_xmax)+'\n')
        f.write('y [min] : '+str(TopFrame.re_ymin)+'\n')
        f.write('y [max] : '+str(TopFrame.re_ymax)+'\n')
        f.write('\n')

        f.write('# Masking Parameters\n')
        f.write('Ellipse 1 center x : '+str(TopFrame.mx1)+'\n')
        f.write('Ellipse 1 center y : '+str(TopFrame.my1)+'\n')
        f.write('Ellipse 1 major (a): '+str(TopFrame.ma1)+'\n')
        f.write('Ellipse 1 minor (b): '+str(TopFrame.mb1)+'\n')
        f.write('Ellipse 1 PA : '+str(TopFrame.mt1)+'\n')
        f.write('\n')
        f.write('Ellipse 2 center x : '+str(TopFrame.mx2)+'\n')
        f.write('Ellipse 2 center y : '+str(TopFrame.my2)+'\n')
        f.write('Ellipse 2 major (a): '+str(TopFrame.ma2)+'\n')
        f.write('Ellipse 2 minor (b): '+str(TopFrame.mb2)+'\n')
        f.write('Ellipse 2 PA : '+str(TopFrame.mt2)+'\n')
        f.write('\n')

        f.write('# Suggested Shift'+'\n')
        f.write('# Note : Relative shift of the high frequency map with respect to the low frequency map.'+'\n')
        f.write('Shift in x [pix]   : '+str(TopFrame.shift_x)+'\n')
        f.write('Shift in y [pix]   : '+str(TopFrame.shift_y)+'\n')
        f.write('Corr. value at the peak : '+str(TopFrame.cor_val)+'\n')
        f.write('Total shift [mas]  : '+str(TopFrame.shift_mas)+'\n')
        f.write('Pixel size [mas] : ' + str(TopFrame.dxmas) + ' \n')
        f.write('Pixel size [mas]: ' + str(TopFrame.dymas) + '\n')
        f.write('\n')

        f.write('# SPIX Display Parameters'+'\n')
        f.write('Cutoff by Flux [% of the stokes I peak] : '+str(self.spix_cut*1e-1)+'\n')
        f.write('Cutoff by SPIX Error [no unit] : '+str(self.spix_err_limit)+'\n')
        f.write('RMS 1 [mJy/bm] : '+str(1e+3*self.rms1)+'\n')
        f.write('RMS 2 [mJy/bm] : '+str(1e+3*self.rms2)+'\n')
        f.write('Calib err 1 [%] : '+str(1e+2*self.cal_err1)+'\n')
        f.write('Calib err 2 [%] : '+str(1e+2*self.cal_err2)+'\n')
        f.write('Manual shift (if used) in x [pix]: '+str(self.shift_x)+'\n')
        f.write('Manual shift (if used) in y [pix]: '+str(self.shift_y)+'\n')

        f.write('#'*30)
        f.write('\n# End of output file\n')
        f.write('#'*30)

        f.close()

        print('Saved output at : '+' "VIMAP_output'+str(self.output_v)+'.dat"')

    def OnEXIT(self, event):
        self.Close()







class MyApp(wx.App):
    def OnInit(self):
        top_frame = TopFrame(None, -1, 'VIMAP version '+str(ver))
        top_frame.Show(True)
        return True

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    app = MyApp()
    app.MainLoop()
