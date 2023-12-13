// Code generated by "goki generate"; DO NOT EDIT.

package main

import (
	"goki.dev/gti"
	"goki.dev/ordmap"
)

var _ = gti.AddType(&gti.Type{
	Name:      "main.V1Img",
	ShortName: "main.V1Img",
	IDName:    "v-1-img",
	Doc:       "Img manages conversion of a bitmap image into tensor formats for\nsubsequent processing by filters.",
	Directives: gti.Directives{
		&gti.Directive{Tool: "gti", Directive: "add", Args: []string{}},
	},
	Fields: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
		{"File", &gti.Field{Name: "File", Type: "goki.dev/gi/v2/gi.FileName", LocalType: "gi.FileName", Doc: "name of image file to operate on", Directives: gti.Directives{}, Tag: ""}},
		{"Size", &gti.Field{Name: "Size", Type: "image.Point", LocalType: "image.Point", Doc: "target image size to use -- images will be rescaled to this size", Directives: gti.Directives{}, Tag: ""}},
		{"Img", &gti.Field{Name: "Img", Type: "image.Image", LocalType: "image.Image", Doc: "current input image", Directives: gti.Directives{}, Tag: "view:\"-\""}},
		{"Tsr", &gti.Field{Name: "Tsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "input image as an RGB tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"LMS", &gti.Field{Name: "LMS", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "LMS components + opponents tensor version of image", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
	}),
	Embeds: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}),
	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{
		{"OpenImage", &gti.Method{Name: "OpenImage", Doc: "OpenImage opens given filename as current image Img\nand converts to a float32 tensor for processing", Directives: gti.Directives{
			&gti.Directive{Tool: "gti", Directive: "add", Args: []string{}},
		}, Args: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
			{"filepath", &gti.Field{Name: "filepath", Type: "string", LocalType: "string", Doc: "", Directives: gti.Directives{}, Tag: ""}},
			{"filtsz", &gti.Field{Name: "filtsz", Type: "int", LocalType: "int", Doc: "", Directives: gti.Directives{}, Tag: ""}},
		}), Returns: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
			{"error", &gti.Field{Name: "error", Type: "error", LocalType: "error", Doc: "", Directives: gti.Directives{}, Tag: ""}},
		})}},
	}),
})

var _ = gti.AddType(&gti.Type{
	Name:      "main.V1sOut",
	ShortName: "main.V1sOut",
	IDName:    "v-1-s-out",
	Doc:       "V1sOut contains output tensors for V1 Simple filtering, one per opponnent",
	Directives: gti.Directives{
		&gti.Directive{Tool: "gti", Directive: "add", Args: []string{}},
	},
	Fields: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
		{"Tsr", &gti.Field{Name: "Tsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"ExtGiTsr", &gti.Field{Name: "ExtGiTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple extra Gi from neighbor inhibition tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"KwtaTsr", &gti.Field{Name: "KwtaTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output, kwta output tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"PoolTsr", &gti.Field{Name: "PoolTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output, max-pooled 2x2 of Kwta tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
	}),
	Embeds:  ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}),
	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{}),
})

var _ = gti.AddType(&gti.Type{
	Name:      "main.Vis",
	ShortName: "main.Vis",
	IDName:    "vis",
	Doc:       "Vis encapsulates specific visual processing pipeline in\nuse in a given case -- can add / modify this as needed.\nHandles 3 major opponent channels: WhiteBlack, RedGreen, BlueYellow",
	Directives: gti.Directives{
		&gti.Directive{Tool: "gti", Directive: "add", Args: []string{}},
	},
	Fields: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
		{"Color", &gti.Field{Name: "Color", Type: "bool", LocalType: "bool", Doc: "if true, do full color filtering -- else Black/White only", Directives: gti.Directives{}, Tag: ""}},
		{"SepColor", &gti.Field{Name: "SepColor", Type: "bool", LocalType: "bool", Doc: "record separate rows in V1s summary for each color -- otherwise just records the max across all colors", Directives: gti.Directives{}, Tag: ""}},
		{"ColorGain", &gti.Field{Name: "ColorGain", Type: "float32", LocalType: "float32", Doc: "extra gain for color channels -- lower contrast in general", Directives: gti.Directives{}, Tag: "def:\"8\""}},
		{"Img", &gti.Field{Name: "Img", Type: "*github.com/emer/vision/v2/examples/color_gabor.V1Img", LocalType: "*V1Img", Doc: "image that we operate upon -- one image often shared among multiple filters", Directives: gti.Directives{}, Tag: ""}},
		{"V1sGabor", &gti.Field{Name: "V1sGabor", Type: "github.com/emer/vision/v2/gabor.Filter", LocalType: "gabor.Filter", Doc: "V1 simple gabor filter parameters", Directives: gti.Directives{}, Tag: ""}},
		{"V1sGeom", &gti.Field{Name: "V1sGeom", Type: "github.com/emer/vision/v2/vfilter.Geom", LocalType: "vfilter.Geom", Doc: "geometry of input, output for V1 simple-cell processing", Directives: gti.Directives{}, Tag: "inactive:\"+\" view:\"inline\""}},
		{"V1sNeighInhib", &gti.Field{Name: "V1sNeighInhib", Type: "github.com/emer/vision/v2/kwta.NeighInhib", LocalType: "kwta.NeighInhib", Doc: "neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code", Directives: gti.Directives{}, Tag: ""}},
		{"V1sKWTA", &gti.Field{Name: "V1sKWTA", Type: "github.com/emer/vision/v2/kwta.KWTA", LocalType: "kwta.KWTA", Doc: "kwta parameters for V1s", Directives: gti.Directives{}, Tag: ""}},
		{"V1sGaborTsr", &gti.Field{Name: "V1sGaborTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sGaborTab", &gti.Field{Name: "V1sGaborTab", Type: "goki.dev/etable/v2/etable.Table", LocalType: "etable.Table", Doc: "V1 simple gabor filter table (view only)", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1s", &gti.Field{Name: "V1s", Type: "[3]github.com/emer/vision/v2/examples/color_gabor.V1sOut", LocalType: "[colorspace.OpponentsN]V1sOut", Doc: "V1 simple gabor filter output, per channel", Directives: gti.Directives{}, Tag: "view:\"inline\""}},
		{"V1sMaxTsr", &gti.Field{Name: "V1sMaxTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "max over V1 simple gabor filters output tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sPoolTsr", &gti.Field{Name: "V1sPoolTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output, max-pooled 2x2 of Kwta tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sUnPoolTsr", &gti.Field{Name: "V1sUnPoolTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output, un-max-pooled 2x2 of Pool tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"ImgFmV1sTsr", &gti.Field{Name: "ImgFmV1sTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "input image reconstructed from V1s tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sAngOnlyTsr", &gti.Field{Name: "V1sAngOnlyTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output, angle-only features tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sAngPoolTsr", &gti.Field{Name: "V1sAngPoolTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1cLenSumTsr", &gti.Field{Name: "V1cLenSumTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 complex length sum filter output tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1cEndStopTsr", &gti.Field{Name: "V1cEndStopTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "V1 complex end stop filter output tensor", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1AllTsr", &gti.Field{Name: "V1AllTsr", Type: "goki.dev/etable/v2/etensor.Float32", LocalType: "etensor.Float32", Doc: "Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total (9 if SepColor)", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
		{"V1sInhibs", &gti.Field{Name: "V1sInhibs", Type: "github.com/emer/vision/v2/fffb.Inhibs", LocalType: "fffb.Inhibs", Doc: "inhibition values for V1s KWTA", Directives: gti.Directives{}, Tag: "view:\"no-inline\""}},
	}),
	Embeds: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}),
	Methods: ordmap.Make([]ordmap.KeyVal[string, *gti.Method]{
		{"Filter", &gti.Method{Name: "Filter", Doc: "Filter is overall method to run filters on current image file name\nloads the image from ImageFile and then runs filters", Directives: gti.Directives{
			&gti.Directive{Tool: "gti", Directive: "add", Args: []string{}},
		}, Args: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{}), Returns: ordmap.Make([]ordmap.KeyVal[string, *gti.Field]{
			{"error", &gti.Field{Name: "error", Type: "error", LocalType: "error", Doc: "", Directives: gti.Directives{}, Tag: ""}},
		})}},
	}),
})
