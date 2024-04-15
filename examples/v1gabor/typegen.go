// Code generated by "core generate -add-types"; DO NOT EDIT.

package main

import (
	"cogentcore.org/core/types"
)

var _ = types.AddType(&types.Type{Name: "main.Vis", IDName: "vis", Doc: "Vis encapsulates specific visual processing pipeline in\nuse in a given case -- can add / modify this as needed", Directives: []types.Directive{{Tool: "types", Directive: "add"}}, Methods: []types.Method{{Name: "OpenImage", Doc: "OpenImage opens given filename as current image Img\nand converts to a float32 tensor for processing", Directives: []types.Directive{{Tool: "types", Directive: "add"}}, Args: []string{"filepath"}, Returns: []string{"error"}}, {Name: "Filter", Doc: "Filter is overall method to run filters on current image file name\nloads the image from ImageFile and then runs filters", Directives: []types.Directive{{Tool: "types", Directive: "add"}}, Returns: []string{"error"}}}, Fields: []types.Field{{Name: "ImageFile", Doc: "name of image file to operate on"}, {Name: "V1sGabor", Doc: "V1 simple gabor filter parameters"}, {Name: "V1sGeom", Doc: "geometry of input, output for V1 simple-cell processing"}, {Name: "V1sNeighInhib", Doc: "neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code"}, {Name: "V1sKWTA", Doc: "kwta parameters for V1s"}, {Name: "ImgSize", Doc: "target image size to use -- images will be rescaled to this size"}, {Name: "V1sGaborTsr", Doc: "V1 simple gabor filter tensor"}, {Name: "V1sGaborTab", Doc: "V1 simple gabor filter table (view only)"}, {Name: "Img", Doc: "current input image"}, {Name: "ImgTsr", Doc: "input image as tensor"}, {Name: "ImgFromV1sTsr", Doc: "input image reconstructed from V1s tensor"}, {Name: "V1sTsr", Doc: "V1 simple gabor filter output tensor"}, {Name: "V1sExtGiTsr", Doc: "V1 simple extra Gi from neighbor inhibition tensor"}, {Name: "V1sKwtaTsr", Doc: "V1 simple gabor filter output, kwta output tensor"}, {Name: "V1sPoolTsr", Doc: "V1 simple gabor filter output, max-pooled 2x2 of V1sKwta tensor"}, {Name: "V1sUnPoolTsr", Doc: "V1 simple gabor filter output, un-max-pooled 2x2 of V1sPool tensor"}, {Name: "V1sAngOnlyTsr", Doc: "V1 simple gabor filter output, angle-only features tensor"}, {Name: "V1sAngPoolTsr", Doc: "V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor"}, {Name: "V1cLenSumTsr", Doc: "V1 complex length sum filter output tensor"}, {Name: "V1cEndStopTsr", Doc: "V1 complex end stop filter output tensor"}, {Name: "V1AllTsr", Doc: "Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total"}, {Name: "V1sInhibs", Doc: "inhibition values for V1s KWTA"}}})