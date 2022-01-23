#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:53:01 2022

@author: dan
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib

from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator,
                                                 DictFormatter)
import matplotlib.pyplot as plt

matplotlib.pyplot.close('all')

#generate random data np uni
# xVals=np.random.uniform(low=0, high=np.pi*.5, size=100)
# yVals=np.random.uniform(low=1, high=2, size=100)
xVals=np.random.rand(1000)
yVals=np.random.rand(1000)
xRange=[0,1]
yRange=[0,1]


#create an array
sampleData=np.asarray([xVals,yVals]).T
#populate dataframe
sampleDataFrame=pd.DataFrame(data=sampleData,columns=['xVals','yVals'])
#initialize figure and axis
#fig, ax = matplotlib.pyplot.subplots()
fig=matplotlib.pyplot.figure(figsize=(8, 8))

#get the dimensions of the figure
figDims=fig.get_size_inches()
#find the center coordinate
figCenterCoord=figDims/2

ringNumber=2
#establish Figure Number
figureNumber=5
#find radians dedicated to each
spaceForEach=(2*np.pi)/figureNumber
#create boundaries in radian space
boundaries=np.arange(0,2*np.pi,spaceForEach)

def transformDataToWedgeRange(sampleDataFrame, xLabel, yLabel,figureNumber,ringNumber,forceBounds=None):
    import numpy as np

    
    spaceForEach=(2*np.pi)/figureNumber
    #create boundaries in radian space
    boundaries=np.arange(0,2*np.pi,spaceForEach)
    
    if forceBounds==None:
        sampleDataFrame[xLabel]=(sampleDataFrame[xLabel]-np.min(sampleDataFrame[xLabel])/np.max(sampleDataFrame[xLabel]))
        sampleDataFrame[xLabel]=sampleDataFrame[xLabel]*[boundaries[1]]
        sampleDataFrame[yLabel]=(sampleDataFrame[yLabel]-np.min(sampleDataFrame[yLabel])/np.max(sampleDataFrame[yLabel]))
    
    else:
        #essentially normalize the data
        sampleDataFrame[xLabel]=(sampleDataFrame[xLabel]-forceBounds[0])/forceBounds[1]
        sampleDataFrame[xLabel]=sampleDataFrame[xLabel]*[boundaries[1]]
        sampleDataFrame[yLabel]=(sampleDataFrame[yLabel]-forceBounds[2])/forceBounds[3]+ringNumber-1
    
    return sampleDataFrame

def computeaAxesBoundingBoxes(fig, figureNumber,ringNumber):
    import numpy as np
    
    boundingBoxCoords=np.zeros([2,2,figureNumber])
    
    #get the dimensions of the figure
    figDims=fig.get_size_inches()
    #find the center coordinate
    figCenterCoord=figDims/2
    
    spaceForEach=(2*np.pi)/figureNumber
    #create boundaries in radian space
    boundaries=np.arange(0,2*np.pi,spaceForEach)
    boundaries=np.append(boundaries,boundaries[-1]+spaceForEach)
    
    
    

    
    for isubFigs in range(figureNumber):
        corner1=pol2cart(ringNumber-1,boundaries[isubFigs])
        corner2=pol2cart(ringNumber-1,boundaries[isubFigs +1])
        corner3=pol2cart(ringNumber,boundaries[isubFigs])
        corner4=pol2cart(ringNumber,boundaries[isubFigs +1])
        xMin=np.min([[corner1[0],corner2[0]],[corner3[0],corner4[0]]])
        xMax=np.max([[corner1[0],corner2[0]],[corner3[0],corner4[0]]])
        yMin=np.min([[corner1[1],corner2[1]],[corner3[1],corner4[1]]])
        yMax=np.max([[corner1[1],corner2[1]],[corner3[1],corner4[1]]])
        boundingBoxCoords[:,:,isubFigs]=[[xMin,xMax],[yMin,yMax]]
    
    shiftedScaledBBoxes=(boundingBoxCoords+figCenterCoord[0])/figDims[0]
    
    return shiftedScaledBBoxes

def computeArcBoundingBox(fig,arcBound1,arcBound2,radius1,radius2):
    import numpy as np
    
    figCenterCoord=fig.get_size_inches()/2
    
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    cardinalPoints=np.arange(0,(2*np.pi),(2*np.pi)/4)
    
    cardinalCoords1=np.asarray([pol2cart(radius1, iPoints) for iPoints in cardinalPoints])
    cardinalCoords2=np.asarray([pol2cart(radius2, iPoints) for iPoints in cardinalPoints])
    
    cardinalBoundsCheck=[ iCardinalPoints>arcBound1 and iCardinalPoints<arcBound2  for iCardinalPoints in cardinalPoints ]
    
    arcCoords11=pol2cart(radius1, arcBound1)
    arcCoords12=pol2cart(radius1, arcBound2)
    arcCoords21=pol2cart(radius2, arcBound1)
    arcCoords22=pol2cart(radius2, arcBound2)
    
    arcCoordsArray=np.vstack((arcCoords11,arcCoords12,arcCoords21,arcCoords22))
    
    coordsToCheck=np.vstack((cardinalCoords1[cardinalBoundsCheck],cardinalCoords2[cardinalBoundsCheck],arcCoordsArray))
    
    xMin=np.min(coordsToCheck[:,0])
    xMax=np.max(coordsToCheck[:,0])
    yMin=np.min(coordsToCheck[:,1])
    yMax=np.max(coordsToCheck[:,1])
    
    return [xMin,  xMax, yMin, yMax]
    

    
#    fig=matplotlib.pyplot.figure()
transformedData=transformDataToWedgeRange(sampleDataFrame, xLabel, yLabel,figureNumber,ringNumber,forceBounds)


#warp the axis?
fig=matplotlib.pyplot.figure(figsize=(8, 8))

# seabornStyleDict=sns.axes_style()
# seabornStyleDict['xtick.bottom']=False
# seabornStyleDict['ytick.left']=False
for iBoundaries in range(5):
    dataBounds = [boundaries[iBoundaries], boundaries[iBoundaries+1], 2, 4]
    print(iBoundaries)
    #ax1, aux_ax=setup_axes2(fig ,111, dataBounds, Affine2D().rotate_deg(np.degrees(iBoundaries)))
    ax1, aux_ax=setup_axes2(fig, dataBounds)
    
    #set scatterplot into axis / figure
    sns.kdeplot(data=transformedData, x='xVals', y='yVals',ax=ax1,fill=True)
    



def setup_axes2(fig, dataBounds, transform=None ):
    """
    

    Parameters
    ----------
    fig : figure handle
        DESCRIPTION.
    rect : subplot figure location
        DESCRIPTION.

    Returns
    -------
    ax1 : TYPE
        DESCRIPTION.
    aux_ax : TYPE
        DESCRIPTION.

    """
    #setup transform method, polar transform
    if transform==None:
        tr = PolarAxes.PolarTransform()
    else:
        tr = transform + PolarAxes.PolarTransform()
    #get pi from numpy
    pi = np.pi
    
    #create the tic labels that will be used
    # angle_ticks = [(0, r"$0$"),
    #                (.25*pi, r"$\frac{1}{4}\pi$"),
    #                (.5*pi, r"$\frac{1}{2}\pi$")]
    # grid_locator1 = FixedLocator([v for v, s in angle_ticks])
    # tick_formatter1 = DictFormatter(dict(angle_ticks))

    # grid_locator2 = MaxNLocator(2)
    
    #mpl_toolkits.axisartist.floating_axes.GridHelperCurveLinear(aux_trans, 
    #extremes, grid_locator1=None, grid_locator2=None, tick_formatter1=None,
    #tick_formatter2=None)
    #
    bboxCoords=computeArcBoundingBox(fig,dataBounds[0],dataBounds[1],radius1=dataBounds[2],radius2=dataBounds[3])

    figCenterCoord=fig.get_size_inches()/2
    shiftedScaledBBoxes=(bboxCoords+figCenterCoord[0])/fig.get_size_inches()[0]
    
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=( dataBounds[0], dataBounds[1], dataBounds[3], dataBounds[2]),
        grid_locator1=None,
        grid_locator2=None,
        tick_formatter1=None,
        tick_formatter2=None)

    ax1 = fig.add_subplot(111, axes_class=floating_axes.FloatingAxes, grid_helper=grid_helper)
    
    ax1.axis["top"].toggle(all=False)
    ax1.axis["bottom"].toggle(all=False)
    ax1.axis["left"].toggle(all=False)
    ax1.axis["right"].toggle(all=False)
    print(str(shiftedScaledBBoxes))
    currentBBox=matplotlib.transforms.Bbox.from_extents(shiftedScaledBBoxes[0],shiftedScaledBBoxes[2],shiftedScaledBBoxes[1],shiftedScaledBBoxes[3])
    
    ax1.set_position(currentBBox)
    ax1.margins(0,tight=True)

    
    
    #ax1.axis["top"].major_ticklabels.set_axis_direction("top")

    # create a parasite axes whose transData in RA, cz
    #what the heck is a parasite axis?
    aux_ax = ax1.get_aux_axes(tr)
    aux_ax.set_position(currentBBox)
    aux_ax.margins(0,tight=True)

    #aux_ax.set_visible(False)

    aux_ax.patch = ax1.patch  # for aux_ax to have a clip path as in ax
    ax1.patch.zorder = 0.9  # but this has a side effect that the patch is
    # drawn twice, and possibly over some other
    # artists. So, we decrease the zorder a bit to
    # prevent this.
    
   
    return ax1, aux_ax





#now we try warping?

#get rcParams
rcParams = matplotlib.rcParams