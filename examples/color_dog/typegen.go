// Code generated by "core generate -add-types"; DO NOT EDIT.

package main

import (
	"cogentcore.org/core/types"
)

var _ = types.AddType(&types.Type{Name: "main.Vis", IDName: "vis", Doc: "Vis encapsulates specific visual processing pipeline in\nuse in a given case -- can add / modify this as needed", Directives: []types.Directive{{Tool: "types", Directive: "add"}}, Methods: []types.Method{{Name: "OpenImage", Doc: "OpenImage opens given filename as current image Img", Directives: []types.Directive{{Tool: "types", Directive: "add"}}, Args: []string{"filepath"}, Returns: []string{"error"}}, {Name: "Filter", Doc: "Filter is overall method to run filters on current image file name\nloads the image from ImageFile and then runs filters", Directives: []types.Directive{{Tool: "types", Directive: "add"}}, Returns: []string{"error"}}}, Fields: []types.Field{{Name: "ImageFile", Doc: "name of image file to operate on -- if macbeth or empty use the macbeth standard color test image"}, {Name: "DoG", Doc: "LGN DoG filter parameters"}, {Name: "DoGNames", Doc: "names of the dog gain sets -- for naming output data"}, {Name: "DoGGains", Doc: "overall gain factors, to compensate for diffs in OnGains"}, {Name: "DoGOnGains", Doc: "OnGain factors -- 1 = perfect balance, otherwise has relative imbalance for capturing main effects"}, {Name: "Geom", Doc: "geometry of input, output"}, {Name: "ImgSize", Doc: "target image size to use -- images will be rescaled to this size"}, {Name: "DoGTsr", Doc: "DoG filter tensor -- has 3 filters (on, off, net)"}, {Name: "DoGTab", Doc: "DoG filter table (view only)"}, {Name: "Img", Doc: "current input image"}, {Name: "ImgTsr", Doc: "input image as RGB tensor"}, {Name: "ImgLMS", Doc: "LMS components + opponents tensor version of image"}, {Name: "OutAll", Doc: "output from 3 dogs with different tuning"}, {Name: "OutTsrs", Doc: "DoG filter output tensors"}}})
