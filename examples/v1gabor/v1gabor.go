// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate core generate -add-types

import (
	"image"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/core/core"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	_ "cogentcore.org/lab/gosl/slbool/slboolcore" // include to get gui views
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorcore"
	_ "cogentcore.org/lab/tensorcore" // include to get gui views
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/v1vision/gabor"
	"github.com/emer/v1vision/kwta"
	"github.com/emer/v1vision/v1vision"
)

func main() {
	vi := &Vis{}
	vi.Defaults()
	vi.Config()
	vi.Filter()
	vi.ConfigGUI()
}

// Vis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed
type Vis struct { //types:add
	// GPU means use gpu.
	GPU bool

	// name of image file to operate on
	ImageFile core.Filename

	// V1 simple gabor filter parameters
	V1sGabor gabor.Filter

	// geometry of input, output for V1 simple-cell processing
	V1sGeom v1vision.Geom `edit:"-"`

	// neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code
	V1sNeighInhib kwta.NeighInhib

	// kwta parameters for V1s
	V1sKWTA kwta.KWTA

	// target image size to use -- images will be rescaled to this size
	ImageSize image.Point

	// V1 simple gabor filter tensor
	V1sGaborTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter table (view only)
	V1sGaborTab table.Table `display:"no-inline"`

	// current input image
	Image image.Image `display:"-"`

	// V1 is the V1Vision filter processing system
	V1 v1vision.V1Vision `display:"no-inline"`

	// input image as tensor
	ImageTsr *tensor.Float32 `display:"no-inline"`

	// input image reconstructed from V1s tensor
	ImageFromV1sTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output tensor
	V1sTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, kwta output tensor
	V1sKwtaTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, max-pooled 2x2 of V1sKwta tensor
	V1sPoolTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, un-max-pooled 2x2 of V1sPool tensor
	V1sUnPoolTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, angle-only features tensor
	V1sAngOnlyTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor
	V1sAngPoolTsr tensor.Float32 `display:"no-inline"`

	// V1 complex length sum filter output tensor
	V1cLenSumTsr tensor.Float32 `display:"no-inline"`

	// V1 complex end stop filter output tensor
	V1cEndStopTsr tensor.Float32 `display:"no-inline"`

	// Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total
	V1AllTsr tensor.Float32 `display:"no-inline"`

	tabView *core.Tabs
}

func (vi *Vis) Defaults() {
	vi.GPU = true
	vi.ImageFile = core.Filename("side-tee-128.png")
	vi.V1sGabor.Defaults()
	sz := 12 // V1mF16 typically = 12, no border
	spc := 4
	vi.V1sGabor.SetSize(sz, spc)
	vi.ImageSize = image.Point{128, 128}
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	vi.V1sGeom.SetImage(math32.Vec2i(0, 0), math32.Vec2i(spc, spc), math32.Vec2i(sz, sz), vi.ImageSize)
	vi.V1sNeighInhib.Defaults()
	vi.V1sKWTA.Defaults()
	// vi.V1sKWTA.Layer.On.SetBool(true)
	// vi.V1sKWTA.Pool.On.SetBool(false)
}

// Config sets up the V1 processing pipeline.
func (vi *Vis) Config() {
	vi.V1.Init()
	kw := vi.V1.NewKWTAParams()
	*kw = vi.V1sKWTA
	img := vi.V1.NewImage(vi.V1sGeom.In.V())
	wrap := vi.V1.NewImage(vi.V1sGeom.In.V())
	vi.ImageTsr = vi.V1.Images.SubSpace(0).(*tensor.Float32)
	vi.V1.NewWrapImage(img, 0, wrap, int(vi.V1sGeom.FilterRt.X), &vi.V1sGeom)

	// V1s simple
	_, out := vi.V1.AddGabor(wrap, &vi.V1sGabor, &vi.V1sGeom)
	v1out := out
	if vi.V1sKWTA.On.IsTrue() {
		if vi.V1sNeighInhib.On {
			vi.V1.NewNeighInhib(out, vi.V1sGabor.NAngles, vi.V1sNeighInhib.Gi, &vi.V1sGeom)
		} else {
			vi.V1.NewNeighInhibOutput(vi.V1sGabor.NAngles, &vi.V1sGeom) // blank
		}
		v1out = vi.V1.NewKWTA(out, vi.V1sGabor.NAngles, &vi.V1sGeom)
	}
	_ = v1out

	// V1c complex
	// var mpGeom v1vision.Geom
	// mpGeom.SetFilter(math32.Vec2i(0, 0), math32.Vec2i(2, 2), math32.Vec2i(2, 2), vi.V1sGeom.Out.V())
	// omax := vi.V1.NewMaxPool(out, vi.V1sGabor.NAngles, &mpGeom)

	vi.V1.SetAsCurrent()
	if vi.GPU {
		vi.V1.GPUInit()
	}
}

// OpenImage opens given filename as current image Image
// and converts to a float32 tensor for processing
func (vi *Vis) OpenImage(filepath string) error { //types:add
	var err error
	vi.Image, _, err = imagex.Open(filepath)
	if err != nil {
		return errors.Log(err)
	}
	isz := vi.Image.Bounds().Size()
	if isz != vi.ImageSize {
		vi.Image = transform.Resize(vi.Image, vi.ImageSize.X, vi.ImageSize.Y, transform.Linear)
	}
	img := vi.ImageTsr.SubSpace(0).(*tensor.Float32)
	v1vision.RGBToGrey(vi.Image, img, int(vi.V1sGeom.FilterRt.X), false) // pad for filt, bot zero
	return nil
}

// ImageFromV1Simple reverses V1Simple Gabor filtering from V1s back to input image
func (vi *Vis) ImageFromV1Simple() {
	// tensor.SetShapeFrom(&vi.V1sUnPoolTsr, &vi.V1sTsr)
	// vi.V1sUnPoolTsr.SetZeros()
	// tensor.SetShapeFrom(&vi.ImageFromV1sTsr, &vi.ImageTsr)
	// vi.ImageFromV1sTsr.SetZeros()
	// v1vision.UnPool(math32.Vec2(2, 2), math32.Vec2(2, 2), &vi.V1sUnPoolTsr, &vi.V1sPoolTsr, true)
	// v1vision.Deconv(&vi.V1sGeom, &vi.V1sGaborTsr, &vi.ImageFromV1sTsr, &vi.V1sUnPoolTsr, vi.V1sGabor.Gain)
	// stats.UnitNormOut(&vi.ImageFromV1sTsr, &vi.ImageFromV1sTsr)
}

// V1Complex runs V1 complex filters on top of V1Simple features.
// it computes Angle-only, max-pooled version of V1Simple inputs.
func (vi *Vis) V1Complex() {
	// v1vision.MaxPool(math32.Vec2(2, 2), math32.Vec2(2, 2), &vi.V1sKwtaTsr, &vi.V1sPoolTsr)
	// v1vision.MaxReduceFilterY(&vi.V1sKwtaTsr, &vi.V1sAngOnlyTsr)
	// v1vision.MaxPool(math32.Vec2(2, 2), math32.Vec2(2, 2), &vi.V1sAngOnlyTsr, &vi.V1sAngPoolTsr)
	// v1complex.LenSum4(&vi.V1sAngPoolTsr, &vi.V1cLenSumTsr)
	// v1complex.EndStop4(&vi.V1sAngPoolTsr, &vi.V1cLenSumTsr, &vi.V1cEndStopTsr)
}

// V1All aggregates all the relevant simple and complex features
// into the V1AllTsr which is used for input to a network
func (vi *Vis) V1All() {
	// ny := vi.V1sPoolTsr.DimSize(0)
	// nx := vi.V1sPoolTsr.DimSize(1)
	// nang := vi.V1sPoolTsr.DimSize(3)
	// nrows := 5
	// vi.V1AllTsr.SetShapeSizes(ny, nx, nrows, nang)
	// // 1 length-sum
	// v1vision.FeatAgg([]int{0}, 0, &vi.V1cLenSumTsr, &vi.V1AllTsr)
	// // 2 end-stop
	// v1vision.FeatAgg([]int{0, 1}, 1, &vi.V1cEndStopTsr, &vi.V1AllTsr)
	// // 2 pooled simple cell
	// v1vision.FeatAgg([]int{0, 1}, 3, &vi.V1sPoolTsr, &vi.V1AllTsr)
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error { //types:add
	v1vision.UseGPU = vi.GPU
	err := vi.OpenImage(string(vi.ImageFile))
	if err != nil {
		return errors.Log(err)
	}
	vi.V1.Run(v1vision.ValuesVar, v1vision.ImagesVar)

	out := vi.V1.Values.SubSpace(0).(*tensor.Float32)
	vi.V1sTsr.SetShapeSizes(int(vi.V1sGeom.Out.Y), int(vi.V1sGeom.Out.X), 2, vi.V1sGabor.NAngles)
	tensor.CopyFromLargerShape(&vi.V1sTsr, out)

	kout := vi.V1.Values.SubSpace(2).(*tensor.Float32)
	vi.V1sKwtaTsr.SetShapeSizes(int(vi.V1sGeom.Out.Y), int(vi.V1sGeom.Out.X), 2, vi.V1sGabor.NAngles)
	tensor.CopyFromLargerShape(&vi.V1sKwtaTsr, kout)

	// vi.V1Simple()
	// vi.V1Complex()
	// vi.V1All()
	// vi.ImageFromV1Simple()
	return nil
}

func (vi *Vis) ConfigGUI() *core.Body {
	vi.V1sGaborTab.Init()
	// vi.V1sGabor.ToTable(&vi.V1sGaborTab) // note: view only, testing
	tensorcore.AddGridStylerTo(vi.ImageTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(&vi.ImageFromV1sTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(&vi.V1sGaborTab, func(s *tensorcore.GridStyle) {
		s.Size.Min = 16
		s.Range.Set(-0.05, 0.05)
	})

	b := core.NewBody("v1gabor").SetTitle("V1 Gabor Filtering")
	sp := core.NewSplits(b)
	core.NewForm(sp).SetStruct(vi)
	tb := core.NewTabs(sp)
	vi.tabView = tb
	tf, _ := tb.NewTab("Image")
	tensorcore.NewTensorGrid(tf).SetTensor(vi.ImageTsr)
	tf, _ = tb.NewTab("V1s")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.V1sTsr)
	tf, _ = tb.NewTab("V1sKWTA")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.V1sKwtaTsr)

	sp.SetSplits(.3, .7)

	b.AddTopBar(func(bar *core.Frame) {
		core.NewToolbar(bar).Maker(func(p *tree.Plan) {
			tree.Add(p, func(w *core.FuncButton) { w.SetFunc(vi.Filter) })
		})
	})
	b.RunMainWindow()
	return b
}
