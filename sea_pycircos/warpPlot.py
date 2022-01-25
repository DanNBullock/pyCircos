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

def generateTestDataFrame():
    """
    Quick testing function to generate an vertically oriented data frame for
    use with function testing

    Returns
    -------
    sampleDataFrame : pandas dataframe
        A test data frame with 1000 random values between 1 and 0 in the
        'xVals' and 'yVals' columns

    """
    import pandas as pd
    import numpy as np
    xVals=np.random.normal(.5,.2,10000)
    yVals=np.random.normal(.5,.2,10000)

    #create an array
    sampleData=np.asarray([xVals,yVals]).T
    #populate dataframe
    sampleDataFrame=pd.DataFrame(data=sampleData,columns=['xVals','yVals'])
    
    return sampleDataFrame
    

def transformDataToWedgeRange(sampleDataFrame, xLabel, yLabel,lowerRBound,upperRBound,innerRadius,outerRadius,forceBounds=None):
    """
    Converts input dataframe (or at least the specified columns of such) to be 
    proportionally contained within the specified area of the bounds of the
    radially converted plot

    Parameters
    ----------
    sampleDataFrame : pandas dataframe
        The pandas dataframe containing the information that is desired to be
        plotted
    xLabel : string
        The column label assocaiated with the values that are desired to serve
        as the x values in the requested plot
    yLabel : string
        The column label assocaiated with the values that are desired to serve
        as the y values in the requested plot
    lowerRBound : float, between 0 and 2*pi (thus in radians)
        The lesser of the two lateral bounds of the requested plot, as
        instantiated in the cocentric arc segment plot(s)
    upperRBound : float, between 0 and 2*pi (thus in radians)
        The greater of the two lateral bounds of the requested plot, as
        instantiated in the cocentric arc segment plot(s)
    innerRadius : float, in units of the associated figure size (e.g. inches)
        The closer (relative to the figure center) of the two vertical bounds
        of the requested plot, as instantiated in the cocentric arc segment
        plot(s).
    outerRadius : float, in units of the associated figure size (e.g. inches)
        The further (relative to the figure center) of the two vertical bounds
        of the requested plot, as instantiated in the cocentric arc segment
        plot(s).
    forceBounds : array, 4 values in length, floats, optional
        The minimum and maximum boundaries for the data associated with the
        input x and y valus from the dataframe. The default is None.
        Ordering: xMin, xMax, yMin, yMax

    Returns
    -------
    sampleDataFrame : pandas dataframe
        The same input dataframe as entered into the function, however the 
        columns specified in the input xLabel and yLabel variables have been 
        proportionally (and thus in a relation preserving fashion) converted
        into a range which will fit the radial bounds specified in the 
        lowerRBound, upperRBound, innerRadius, and outerRadius variables

    """
    import numpy as np
    import pandas as pd

    #if no forceBounds were entered
    if forceBounds==None:
        #find the value range covered by each dimension of the input 
        #dataframe data
        dfXrange=np.max(sampleDataFrame[xLabel])-np.min(sampleDataFrame[xLabel])
        dfYrange=np.max(sampleDataFrame[yLabel])-np.min(sampleDataFrame[yLabel])
        
        #find the value range covered by the input radial bounds
        inputRadianRange=upperRBound-lowerRBound
        inputRadiusRange=outerRadius-innerRadius
        
        #compute the conversion factors using these values
        xConversionFactor=dfXrange/inputRadianRange
        yConversionFactor=dfYrange/inputRadiusRange
        
        #divide the origional dataframe column by this value to get the converted
        #plot range
        outDataFrame=pd.DataFrame()
        outDataFrame[xLabel][xLabel]=np.add(np.divide(sampleDataFrame[xLabel],xConversionFactor),np.min(sampleDataFrame[xLabel]))
        outDataFrame[yLabel]=np.add(np.divide(sampleDataFrame[yLabel],yConversionFactor),np.min(sampleDataFrame[yLabel]))
    
    else:
        #if force bounds were entered, do essentially the same as above, except
        #don't infer the bounds from the input dataframe values.
        dfXrange=forceBounds[1]-forceBounds[0]
        dfYrange=forceBounds[3]-forceBounds[2]
        
        #find the value range covered by the input radial bounds
        inputRadianRange=upperRBound-lowerRBound
        inputRadiusRange=outerRadius-innerRadius
        
        #compute the conversion factors using these values
        xConversionFactor=dfXrange/inputRadianRange
        yConversionFactor=dfYrange/inputRadiusRange
        
        #divide the origional dataframe column by this value to get the converted
        #plot range
        outDataFrame=pd.DataFrame()
        outDataFrame[xLabel]=np.add(np.divide(sampleDataFrame[xLabel],xConversionFactor),lowerRBound)
        outDataFrame[yLabel]=np.add(np.divide(sampleDataFrame[yLabel],yConversionFactor),innerRadius)
    
    return outDataFrame

def computeArcBoundingBox(fig,arcBound1,arcBound2,radius1,radius2):
    """
    Computes the appropriate bounding box for the specified arc-segment,
    such that the segment can be accurately / reliably placed in a 
    co-centrically oriented arrangment relative to other 

    Parameters
    ----------
    fig : figure handle
        The figure handle of the figure that the arc plot is to be placed on.
    arcBound1 : float, between 0 and 2*pi (thus in radians)
        The lesser of the two lateral bounds of the requested arc section.
    arcBound2 : float, between 0 and 2*pi (thus in radians)
        The greater of the two lateral bounds of the requested arc section.
    radius1 : float, in units of the associated figure size (e.g. inches)
        The closer (relative to the figure center) of the two vertical bounds
        of the requested plot, as instantiated in the cocentric arc segment
        plot(s).
    radius2 : float, in units of the associated figure size (e.g. inches)
        The further (relative to the figure center) of the two vertical bounds
        of the requested plot, as instantiated in the cocentric arc segment
        plot(s).
    Returns
    -------
    list of [xMin,  xMax, yMin, yMax]
        A 4 value list specifying the lateral and vertical boundaries of the
        bounding box which surrounds the arc section specified in the input
        variables.

    """
    import numpy as np
    
    #this assumes that the figure is square, which it should be for a circular
    #plot
    #obtain the center coordinate of the figure and subsequent circle
    figCenterCoord=fig.get_size_inches()/2
    
    #quick confersion function for polar to cartesian coordinates
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    #compute the radian coordinates of the cardinal points of the circle
    cardinalPoints=np.arange(0,(2*np.pi),(2*np.pi)/4)
    
    #compute the cartesian coordinates of the cardinal points for both the
    #inner and outer circle
    cardinalCoords1=np.asarray([pol2cart(radius1, iPoints) for iPoints in cardinalPoints])
    cardinalCoords2=np.asarray([pol2cart(radius2, iPoints) for iPoints in cardinalPoints])
    
    #check to see if the input radian bounds contain one of these cardinal points
    cardinalBoundsCheck=[ iCardinalPoints>arcBound1 and iCardinalPoints<arcBound2  for iCardinalPoints in cardinalPoints ]
    
    #compute the verticies of the arc delineated by the input bound values
    arcCoords11=pol2cart(radius1, arcBound1)
    arcCoords12=pol2cart(radius1, arcBound2)
    arcCoords21=pol2cart(radius2, arcBound1)
    arcCoords22=pol2cart(radius2, arcBound2)
    
    #stack the coordinates of the arc boundaries into an array
    arcCoordsArray=np.vstack((arcCoords11,arcCoords12,arcCoords21,arcCoords22))
    
    #stack the coordinates of the arc boundaries, along with the included 
    #cardinal points (should there be any) into an array
    coordsToCheck=np.vstack((cardinalCoords1[cardinalBoundsCheck],cardinalCoords2[cardinalBoundsCheck],arcCoordsArray))
    
    #check this array for its max and minimum boundaries in both dimensions
    xMin=np.min(coordsToCheck[:,0])
    xMax=np.max(coordsToCheck[:,0])
    yMin=np.min(coordsToCheck[:,1])
    yMax=np.max(coordsToCheck[:,1])
    
    return [xMin, xMax, yMin, yMax]


def setup_ArcedAxes(fig, dataBounds, transform=None ):
    """
    This function establishes the figure axes for a arc section, the figure
    bounds of which are specified in dataBounds 

    Parameters
    ----------
    fig : figure handle
        The figure handle of the figure that the axes will be palced in
    dataBounds : array or list-like of 4 floats,
        These values indicate the boundaries of the arc-shaped figure
        axes which will be generated by this function.
        - The first two values correspond to the lateral boundaries of the
        subplot (in radians)
        - The last two values correspond to the radial boundaries of the
        subplot (units of the source figure size)

    Returns
    -------
    ax1 : subplot axes
        The desired figure axes handle
    aux_ax : auxillary / parasite 
        Also the desired figure axes handle (I don't know what the difference
        is, honestly)

    """
    #setup transform method, polar transform
    if transform==None:
        tr = PolarAxes.PolarTransform()
    else:
        tr = transform + PolarAxes.PolarTransform()
    #get pi from numpy
    pi = np.pi
    
    #compute the coordinates of the bounding box of the current subplot
    bboxCoords=computeArcBoundingBox(fig,dataBounds[0],dataBounds[1],radius1=dataBounds[2],radius2=dataBounds[3])

    #compute the center coordinate of the figure / circle
    figCenterCoord=fig.get_size_inches()/2
    #bounding box coorinates actually have to be entered in as a value between
    #1 and 0, so we need to make a conversion
    #furthermore, the bounding box coordinates are given in a cartesian schema
    #where the center point is 0,0 (and there are negative values)
    #in order to work around this, we add the center figure coord to offset this
    #and then scale by the size of the (presumably square) figure
    shiftedScaledBBoxes=(bboxCoords+figCenterCoord[0])/fig.get_size_inches()[0]
    
    #set up a grid helper entity
    #may need to play around with the ordering of these, depending on what
    #sequence the extreemes are expecting to be entered in
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=( dataBounds[0], dataBounds[1], dataBounds[2], dataBounds[3]),
        grid_locator1=None,
        grid_locator2=None,
        tick_formatter1=None,
        tick_formatter2=None)

    #establish the floating figure axes via subplot
    ax1 = fig.add_subplot(111, axes_class=floating_axes.FloatingAxes, grid_helper=grid_helper)
    
    #turn off all of the tics and everything
    #this can be changed in the production version to be a bit more adaptable
    ax1.axis["top"].toggle(all=False)
    ax1.axis["bottom"].toggle(all=False)
    ax1.axis["left"].toggle(all=False)
    ax1.axis["right"].toggle(all=False)
    
    #find the current bounding box
    currentBBox=matplotlib.transforms.Bbox.from_extents(shiftedScaledBBoxes[0],shiftedScaledBBoxes[2],shiftedScaledBBoxes[1],shiftedScaledBBoxes[3])
    
    #set the axis 
    ax1.set_position(currentBBox)
    #set the margins to none
    ax1.margins(0,tight=True)

    # create a parasite axes with the polar coordinate framework
    #what the heck is a parasite axis?
    aux_ax = ax1.get_aux_axes(tr)
    #set this location to the specified bounding box
    #maybe not necessary as probably inherited from parent axes
    aux_ax.set_position(currentBBox)
    #also reduce the margins to none
    aux_ax.margins(0,tight=True)

    #not sure, but ok
    aux_ax.patch = ax1.patch  # for aux_ax to have a clip path as in ax
    ax1.patch.zorder = 0.9  # but this has a side effect that the patch is
    # drawn twice, and possibly over some other
    # artists. So, we decrease the zorder a bit to
    # prevent this.
    # (as quoted from source matplotlib documentation)
    return ax1, aux_ax

    
#borrow chord diagram from here
#https://github.com/tfardet/mpl_chord_diagram

#begin the test
fig=matplotlib.pyplot.figure(figsize=(10, 10))

#set number of figures to test on ring
figureTestNumber=16
#find radians dedicated to each
spaceForEach=(2*np.pi)/figureTestNumber
#create boundaries in radian space
boundaries=np.arange(0,(2*np.pi)+spaceForEach,spaceForEach)
#generate the rings we'll want
ringBorders=np.arange(2,fig.get_size_inches()[0],2)


for iRingTest in range(len(ringBorders)-1):
    for iBoundaries in range(figureTestNumber):
        dataBounds = [boundaries[iBoundaries], boundaries[iBoundaries+1], ringBorders[iRingTest],ringBorders[iRingTest+1] ]
    
        ax1, aux_ax=setup_ArcedAxes(fig, dataBounds)
        
        testData=generateTestDataFrame()
        forceBounds=[0,1,0,1]
     
        transformedData=transformDataToWedgeRange(testData, 'xVals', 'yVals', lowerRBound=dataBounds[0],upperRBound=dataBounds[1],innerRadius=dataBounds[2],outerRadius=dataBounds[3],forceBounds=forceBounds)
        
        # #test out various plots
        if iRingTest == 0 :
            sns.kdeplot(data=transformedData, x='xVals', y='yVals',ax=aux_ax,fill=True, cmap="rocket")
        elif iRingTest == 1 :    
            sns.histplot(data=transformedData,x='xVals', y='yVals', ax=aux_ax, bins=50, cmap="mako")
        
        elif iRingTest == 2 :
            sns.kdeplot(data=transformedData, x='xVals', y='yVals',ax=aux_ax,fill=True, cmap="cubehelix")
            
        elif iRingTest == 3 :
            sns.scatterplot(data=transformedData,x='xVals', y='yVals', ax=aux_ax)
           
        

fig








