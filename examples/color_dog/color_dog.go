// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate core generate -add-types

import (
	"fmt"
	"image"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/core/base/timer"
	"cogentcore.org/core/core"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorcore"
	_ "cogentcore.org/lab/tensorcore" // include to get gui views
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/v1vision/dog"
	"github.com/emer/v1vision/v1std"
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

	// name of image file to operate on.
	ImageFile core.Filename

	// LGN DoG filter parameters.
	DoG dog.Filter

	// geometry of input, output.
	Geom v1vision.Geom `edit:"-"`

	// target image size to use -- images will be rescaled to this size.
	ImageSize image.Point

	// V1 is the V1Vision filter processing system.
	V1 v1vision.V1Vision `display:"no-inline"`

	// DoG filter table (view only).
	DoGTab table.Table `display:"no-inline"`

	// current input image.
	Image image.Image `display:"-"`

	// input image as tensor.
	ImageTsr *tensor.Float32 `display:"no-inline"`

	// input image as tensor: red, green components (L,M).
	ImageRGTsr *tensor.Float32 `display:"no-inline"`

	// input image as tensor: blue, yellow components (S, LM).
	ImageBYTsr *tensor.Float32 `display:"no-inline"`

	// Out has RG, BY value outputs in 0, 1 feature positions.
	Out tensor.Float32 `display:"no-inline"`

	// DoGGrey is an encapsulated version of this functionality,
	// which we test here for comparison.
	DoGGrey v1std.DoGGrey

	// StdImage manages images for DoGGrey
	StdImage v1std.Image

	tabView *core.Tabs
}

func (vi *Vis) Defaults() {
	vi.GPU = false
	vi.ImageFile = core.Filename("macbeth.png") // GrangerRainbow.png")
	vi.DoGTab.Init()
	vi.DoG.Defaults()
	sz := 12  // V1mF16 typically = 12, no border -- defaults
	spc := 16 // note: not 4; broader blob tuning
	vi.DoG.SetSize(sz, spc)
	vi.DoG.SetSameSigma(0.5) // no spatial component, just pure contrast

	vi.DoG.Gain = 8 // for stronger On tuning: 4.1,On=1.2, Off: 4.4,On=0.833
	vi.DoG.OnGain = 1

	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FilterRt!
	vi.Geom.Set(math32.Vec2i(0, 0), math32.Vec2i(spc, spc), math32.Vec2i(sz, sz))
	vi.ImageSize = image.Point{128, 128}
	// vi.ImageSize = image.Point{256, 256}
	// vi.ImageSize = image.Point{512, 512}
	vi.Geom.SetImageSize(vi.ImageSize)

	vi.DoGGrey.Defaults()
	vi.StdImage.Defaults()
}

// Config sets up the V1 processing pipeline.
func (vi *Vis) Config() {
	vi.V1.Init()
	img := vi.V1.NewImage(vi.Geom.In.V())
	wrap := vi.V1.NewImage(vi.Geom.In.V())
	lmsRG := vi.V1.NewImage(vi.Geom.In.V())
	lmsBY := vi.V1.NewImage(vi.Geom.In.V())

	vi.ImageTsr = vi.V1.Images.SubSpace(img).(*tensor.Float32)
	vi.V1.NewWrapImage(img, 3, wrap, int(vi.Geom.FilterRt.X), &vi.Geom)
	vi.ImageRGTsr = vi.V1.Images.SubSpace(lmsRG).(*tensor.Float32)
	vi.ImageBYTsr = vi.V1.Images.SubSpace(lmsBY).(*tensor.Float32)
	vi.V1.NewLMSComponents(wrap, lmsRG, lmsBY, 5, &vi.Geom)

	out := vi.V1.NewValues(int(vi.Geom.Out.Y), int(vi.Geom.Out.X), 2)
	dogFt := vi.V1.NewDoGOnOff(&vi.DoG, &vi.Geom)

	vi.V1.NewConvolveDiff(lmsRG, v1vision.Red, lmsRG, v1vision.Green, dogFt, 0, 1, out, 0, vi.DoG.Gain, vi.DoG.OnGain, &vi.Geom)
	vi.V1.NewConvolveDiff(lmsBY, v1vision.Blue, lmsBY, v1vision.Yellow, dogFt, 0, 1, out, 1, vi.DoG.Gain, vi.DoG.OnGain, &vi.Geom)

	// _ = out
	// vi.V1.NewLogValues(out, out, 1, 1.0, &vi.Geom)
	// vi.V1.NewNormDiv(v1vision.MaxScalar, out, out, 1, &vi.Geom)

	vi.V1.SetAsCurrent()
	if vi.GPU {
		vi.V1.GPUInit()
	}

	// vi.DoGGrey.Config(vi.StdImage.Size)
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
	img := vi.V1.Images.SubSpace(0).(*tensor.Float32)
	v1vision.RGBToTensor(vi.Image, img, int(vi.Geom.FilterRt.X), v1vision.BottomZero)
	return nil
}

func (vi *Vis) getTsr(idx int, tsr *tensor.Float32, y, x int32) {
	out := vi.V1.Values.SubSpace(idx).(*tensor.Float32)
	tsr.SetShapeSizes(int(y), int(x), 2, 1)
	tensor.CopyFromLargerShape(tsr, out)
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters.
func (vi *Vis) Filter() error { //types:add
	vi.V1.SetAsCurrent()
	v1vision.UseGPU = vi.GPU
	err := vi.OpenImage(string(vi.ImageFile))
	if err != nil {
		return errors.Log(err)
	}
	tmr := timer.Time{}
	tmr.Start()
	for range 1 {
		vi.V1.Run()
		// vi.V1.Run(v1vision.ValuesVar)
		// note: the read sync operation is currently very slow!
		// this needs to be sped up significantly! hopefully with the
		// fix that they are doing for the firefox issue.
		// https://bugzilla.mozilla.org/show_bug.cgi?id=1870699
	}
	tmr.Stop()
	fmt.Println("GPU:", vi.GPU, "Time:", tmr.Total)
	// image = 128: CPU = 333ms, GPU = 198ms
	// image = 256: CPU = 873ms, GPU = 313ms
	// image = 512: CPU = 2.6s,  GPU = 878ms
	vi.V1.Run(v1vision.ValuesVar, v1vision.ImagesVar)

	vi.getTsr(0, &vi.Out, vi.Geom.Out.Y, vi.Geom.Out.X)

	// vi.DoGGrey.RunImage(&vi.StdImage, vi.Image)

	if vi.tabView != nil {
		vi.tabView.Update()
	}
	return nil
}

func (vi *Vis) ConfigGUI() *core.Body {
	vi.DoG.ToTable(&vi.DoGTab) // note: view only, testing
	tensorcore.AddGridStylerTo(vi.ImageTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(vi.ImageRGTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(vi.ImageBYTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(&vi.DoGTab, func(s *tensorcore.GridStyle) {
		s.Size.Min = 16
		s.Range.Set(-0.1, 0.1)
	})

	b := core.NewBody("lgn_dog").SetTitle("LGN DoG Filtering")
	sp := core.NewSplits(b)
	core.NewForm(sp).SetStruct(vi)
	tb := core.NewTabs(sp)
	vi.tabView = tb
	tf, _ := tb.NewTab("Image")
	tensorcore.NewTensorGrid(tf).SetTensor(vi.ImageTsr)
	tf, _ = tb.NewTab("Image RG")
	tensorcore.NewTensorGrid(tf).SetTensor(vi.ImageRGTsr)
	tf, _ = tb.NewTab("Image BY")
	tensorcore.NewTensorGrid(tf).SetTensor(vi.ImageBYTsr)
	tf, _ = tb.NewTab("DoG")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.Out)

	sp.SetSplits(.3, .7)

	b.AddTopBar(func(bar *core.Frame) {
		core.NewToolbar(bar).Maker(func(p *tree.Plan) {
			tree.Add(p, func(w *core.FuncButton) { w.SetFunc(vi.Filter) })
		})
	})
	b.RunMainWindow()
	return b
}
