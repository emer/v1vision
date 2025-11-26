// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate core generate -add-types

import (
	"image"
	"math"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/core/core"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensor/tmath"
	"cogentcore.org/lab/tensorcore"
	_ "cogentcore.org/lab/tensorcore" // include to get gui views
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/v1vision/dog"
	"github.com/emer/v1vision/v1vision"
)

func main() {
	vi := &Vis{}
	vi.Defaults()
	vi.Filter()
	vi.ConfigGUI()
}

// Vis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed
type Vis struct { //types:add

	// name of image file to operate on
	ImageFile core.Filename

	// LGN DoG filter parameters
	DoG dog.Filter

	// geometry of input, output
	Geom v1vision.Geom `edit:"-"`

	// target image size to use -- images will be rescaled to this size
	ImageSize math32.Vector2i

	// DoG filter tensor -- has 3 filters (on, off, net)
	DoGFilter tensor.Float32 `display:"no-inline"`

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
	vi.ImageFile = core.Filename("side-tee-128.png")
	vi.DoGTab.Init()
	vi.DoG.Defaults()
	sz := 12 // V1mF16 typically = 12, no border
	spc := 4
	vi.DoG.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.Geom.Set(math32.Vector2i{0, 0}, math32.Vector2i{spc, spc}, math32.Vector2i{sz, sz})
	vi.ImageSize = math32.Vector2i{128, 128}
	// vi.ImageSize = math32.Vector2i{64, 64}
	vi.DoG.ToTensor(&vi.DoGFilter)
	vi.DoG.ToTable(&vi.DoGTab) // note: view only, testing
	tensorcore.AddGridStylerTo(&vi.ImageTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(&vi.DoGTab, func(s *tensorcore.GridStyle) {
		s.Size.Min = 16
		s.Range.Set(-0.1, 0.1)
	})
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
	v1vision.RGBToGrey(vi.Image, &vi.ImageTsr, vi.Geom.FiltRt.X, false) // pad for filt, bot zero
	v1vision.WrapPad(&vi.ImageTsr, vi.Geom.FiltRt.X)
	return nil
}

// LGNDoG runs DoG filtering on input image
// must have valid Image in place to start.
func (vi *Vis) LGNDoG() {
	flt := vi.DoG.FilterTensor(&vi.DoGFilter, dog.Net)
	v1vision.Conv1(&vi.Geom, flt, &vi.ImageTsr, &vi.OutTsr, vi.DoG.Gain)
	// log norm is generally good it seems for dogs
	n := vi.OutTsr.Len()
	for i := range n {
		vi.OutTsr.SetFloat1D(math.Log(vi.OutTsr.Float1D(i)+1), i)
	}
	mx := stats.Max(tensor.As1D(&vi.OutTsr))
	tmath.DivOut(&vi.OutTsr, mx, &vi.OutTsr)
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error { //types:add
	err := vi.OpenImage(string(vi.ImageFile))
	if err != nil {
		return errors.Log(err)
	}
	vi.LGNDoG()
	return nil
}

func (vi *Vis) ConfigGUI() *core.Body {
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
