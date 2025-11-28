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
	// GPU means use gpu
	GPU bool

	// name of image file to operate on
	ImageFile core.Filename

	// LGN DoG filter parameters
	DoG dog.Filter

	// geometry of input, output
	Geom v1vision.Geom `edit:"-"`

	// target image size to use -- images will be rescaled to this size
	ImageSize image.Point

	// V1 is the V1Vision filter processing system
	V1 v1vision.V1Vision `display:"no-inline"`

	// DoG filter table (view only)
	DoGTab table.Table `display:"no-inline"`

	// current input image
	Image image.Image `display:"-"`

	// input image as tensor
	ImageTsr tensor.Float32 `display:"no-inline"`

	// DoG filter output tensor
	OutTsr tensor.Float32 `display:"no-inline"`
}

func (vi *Vis) Defaults() {
	vi.GPU = true
	vi.ImageFile = core.Filename("side-tee-128.png")
	vi.DoGTab.Init()
	vi.DoG.Defaults()
	sz := 12 // V1mF16 typically = 12, no border
	spc := 4
	vi.DoG.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FilterRt!
	vi.Geom.Set(math32.Vec2i(0, 0), math32.Vec2i(spc, spc), math32.Vec2i(sz, sz))
	vi.ImageSize = image.Point{128, 128}
	// vi.ImageSize = image.Point{64, 64}
	vi.Geom.SetImageSize(vi.ImageSize)
	// fmt.Println(vi.Geom)
}

// Config sets up the V1 processing pipeline.
func (vi *Vis) Config() {
	vi.V1.Init()
	img := vi.V1.NewImage(vi.Geom.In.V())
	wrap := vi.V1.NewImage(vi.Geom.In.V())
	vi.V1.NewWrapImage(img, 0, wrap, int(vi.Geom.FilterRt.X), &vi.Geom)
	_, out := vi.V1.AddDoG(wrap, &vi.DoG, &vi.Geom)
	// _ = out
	vi.V1.NewLogValues(out, out, 1, 1.0, &vi.Geom)
	vi.V1.NewNormDiv(v1vision.MaxScalar, out, out, 1, &vi.Geom)
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
	img := vi.V1.Images.SubSpace(0, 0).(*tensor.Float32)
	v1vision.RGBToGrey(vi.Image, img, int(vi.Geom.FilterRt.X), false) // pad for filt, bot zero
	return nil
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters.
func (vi *Vis) Filter() error { //types:add
	v1vision.UseGPU = vi.GPU
	err := vi.OpenImage(string(vi.ImageFile))
	if err != nil {
		return errors.Log(err)
	}
	tmr := timer.Time{}
	tmr.Start()
	for range 1000 {
		// note: the read sync operation is currently very slow!
		// this needs to be sped up significantly! hopefully with the
		// fix that they are doing for the firefox issue.
		// https://bugzilla.mozilla.org/show_bug.cgi?id=1870699
		vi.V1.Run(v1vision.ValuesVar)
		out := vi.V1.Values.SubSpace(0).(*tensor.Float32)
		vi.OutTsr.SetShapeSizes(out.ShapeSizes()...)
		vi.OutTsr.CopyFrom(out)
	}
	tmr.Stop()
	fmt.Println("GPU:", vi.GPU, "Time:", tmr.Total)

	vi.V1.Run(v1vision.ValuesVar, v1vision.ImagesVar)
	img := vi.V1.Images.SubSpace(1).(*tensor.Float32)
	vi.ImageTsr.SetShapeSizes(img.ShapeSizes()...)
	vi.ImageTsr.CopyFrom(img)
	return nil
}

func (vi *Vis) ConfigGUI() *core.Body {
	vi.DoG.ToTable(&vi.DoGTab) // note: view only, testing
	tensorcore.AddGridStylerTo(&vi.ImageTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(&vi.DoGTab, func(s *tensorcore.GridStyle) {
		s.Size.Min = 16
		s.Range.Set(-0.1, 0.1)
	})

	b := core.NewBody("lgn_dog").SetTitle("LGN DoG Filtering")
	core.NewForm(b).SetStruct(vi)
	b.AddTopBar(func(bar *core.Frame) {
		core.NewToolbar(bar).Maker(func(p *tree.Plan) {
			tree.Add(p, func(w *core.FuncButton) { w.SetFunc(vi.Filter) })
		})
	})
	b.RunMainWindow()
	return b
}
