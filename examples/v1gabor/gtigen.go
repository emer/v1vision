// Code generated by "goki generate"; DO NOT EDIT.

package main

import (
	"goki.dev/gti"
	"goki.dev/ordmap"
)

var _ = gti.AddType(&gti.Type{
	Name:      "main.Vis",
	ShortName: "main.Vis",
	IDName:    "vis",
	Doc:       "Vis encapsulates specific visual processing pipeline in\nuse in a given case -- can add / modify this as needed",
	Directives: gti.Directives{
		&gti.Directive{Tool: "gti", Directive: "add", Args: []string{}},
	},
	Fields: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
		{"ImageFile", &gti.Field{Name: "ImageFile", Type: "goki.dev/gi/v2/gi.FileName", LocalType: "gi.FileName", Doc: "name of image file to operate on", Directives: gti.Directives{}, Tag: ""}},
		{"V1sGabor", &gti.Field{Name: "V1sGabor", Type: "github.com/emer/vision/v2/gabor.Filter", LocalType: "gabor.Filter", Doc: "V1 simple gabor filter parameters", Directives: gti.Directives{}, Tag: ""}},
		{"V1sGeom", &gti.Field{Name: "V1sGeom", Type: "github.com/emer/vision/v2/vfilter.Geom", LocalType: "vfilter.Geom", Doc: "geometry of input, output for V1 simple-cell processing", Directives: gti.Directives{}, Tag: "edit:\"-\""}},
		{"V1sNeighInhib", &gti.Field{Name: "V1sNeighInhib", Type: "github.com/emer/vision/v2/kwta.NeighInhib", LocalType: "kwta.NeighInhib", Doc: "neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code", Directives: gti.Directives{}, Tag: ""}},
		{"V1sKWTA", &gti.Field{Name: "V1sKWTA", Type: "github.com/emer/vision/v2/kwta.KWTA", LocalType: "kwta.KWTA", Doc: "kwta parameters for V1s", Directives: gti.Directives{}, Tag: ""}},
		{"ImgSize", &gti.Field{Name: "ImgSize", Type: "image.Point", LocalType: "image.Point", Doc: "target image size to use -- images will be rescaled to this size", Directives: gti.Directives{}, Tag: ""}},
		{"V1sGaborTsr", &gti.Field{Name: "V1sGaborTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sGaborTab", &gti.Field{Name: "V1sGaborTab", Type: "goki.dev/etable/v2/etable.Table", LocalType: "etable.Table", Doc: "V1 simple gabor filter table (view only)", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"Img", &gti.Field{Name: "Img", Type: "image.Image", LocalType: "image.Image", Doc: "current input image", Directives: gti.Directives{}, Tag: "view:\"-\""}},
		{"ImgTsr", &gti.Field{Name: "ImgTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "input image as tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"ImgFmV1sTsr", &gti.Field{Name: "ImgFmV1sTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "input image reconstructed from V1s tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sTsr", &gti.Field{Name: "V1sTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sExtGiTsr", &gti.Field{Name: "V1sExtGiTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple extra Gi from neighbor inhibition tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sKwtaTsr", &gti.Field{Name: "V1sKwtaTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output, kwta output tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sPoolTsr", &gti.Field{Name: "V1sPoolTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output, max-pooled 2x2 of V1sKwta tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sUnPoolTsr", &gti.Field{Name: "V1sUnPoolTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output, un-max-pooled 2x2 of V1sPool tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sAngOnlyTsr", &gti.Field{Name: "V1sAngOnlyTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output, angle-only features tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sAngPoolTsr", &gti.Field{Name: "V1sAngPoolTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1cLenSumTsr", &gti.Field{Name: "V1cLenSumTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 complex length sum filter output tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1cEndStopTsr", &gti.Field{Name: "V1cEndStopTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 complex end stop filter output tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1AllTsr", &gti.Field{Name: "V1AllTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sInhibs", &gti.Field{Name: "V1sInhibs", Type: "github.com/emer/vision/v2/fffb.Inhibs", LocalType: "fffb.Inhibs", Doc: "inhibition values for V1s KWTA", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
	}),
	Embeds: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}),
	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{
		{"OpenImage", &gti.Method{Name: "OpenImage", Doc: "OpenImage opens given filename as current image Img\nand converts to a float32 tensor for processing", Directives: gti.Directives{
			&gti.Directive{Tool: "gti", Directive: "add", Args: []string{}},
		}, Args: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
			{"filepath", &gti.Field{Name: "filepath", Type: "string", LocalType: "string", Doc: "", Directives: gti.Directives{}, Tag: ""}},
		}), Returns: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
			{"error", &gti.Field{Name: "error", Type: "error", LocalType: "error", Doc: "", Directives: gti.Directives{}, Tag: ""}},
		})}},
		{"Filter", &gti.Method{Name: "Filter", Doc: "Filter is overall method to run filters on current image file name\nloads the image from ImageFile and then runs filters", Directives: gti.Directives{
			&gti.Directive{Tool: "gti", Directive: "add", Args: []string{}},
		}, Args: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}), Returns: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
			{"error", &gti.Field{Name: "error", Type: "error", LocalType: "error", Doc: "", Directives: gti.Directives{}, Tag: ""}},
		})}},
	}),
})
