// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate core generate -add-types

import (
	"image"
	"math"

	"cogentcore.org/core/core"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensor/tmath"
	"cogentcore.org/lab/tensorcore"
	_ "cogentcore.org/lab/tensorcore" // include to get gui views
	"github.com/emer/emergent/v2/edge"
	"github.com/emer/v1vision/dog"
	"github.com/emer/v1vision/motion"
	"github.com/emer/v1vision/vfilter"
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

	// NFrames is the number of frames to render per trial.
	NFrames int

	// target image size to use.
	ImageSize image.Point

	// Bar is the size of the moving bar.
	Bar image.Point

	// Velocity is the motion direction vector.
	Velocity image.Point

	// Start is the starting position.
	Start image.Point

	// Pos is current position
	Pos image.Point `edit:"-"`

	// Motion parameters
	Motion motion.Params

	// LGN DoG filter parameters
	DoG dog.Filter

	// geometry of input, output
	Geom vfilter.Geom `edit:"-"`

	// input image as tensor
	ImageTsr tensor.Float32 `display:"no-inline"`

	// DoG filter tensor -- has 3 filters (on, off, net)
	DoGFilter tensor.Float32 `display:"no-inline"`

	// DoG filter table (view only)
	DoGTab table.Table `display:"no-inline"`

	// DoG filter output tensor
	DoGOutTsr tensor.Float32 `display:"no-inline"`

	Slow      tensor.Float32 `display:"no-inline"`
	Fast      tensor.Float32 `display:"no-inline"`
	MotionOut tensor.Float32 `display:"no-inline"`
}

func (vi *Vis) Defaults() {
	vi.NFrames = 30
	vi.ImageSize = image.Point{128, 128}
	vi.ImageTsr.SetShapeSizes(128, 128)
	vi.Bar = image.Point{8, 16}
	vi.Velocity = image.Point{2, 0}
	vi.Start = image.Point{8, 64}
	vi.DoGTab.Init()
	vi.Motion.Defaults()
	vi.Motion.Gain = 20
	vi.DoG.Defaults()
	sz := 12 // V1mF16 typically = 12, no border
	spc := 4
	vi.DoG.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.Geom.Set(image.Point{0, 0}, image.Point{spc, spc}, image.Point{sz, sz})
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

// RenderFrames renders the frames
func (vi *Vis) RenderFrames() { //types:add
	vi.Pos = vi.Start
	for range vi.NFrames {
		vi.RenderFrame()
		vi.LGNDoG()
		vi.Motion.IntegrateFrame(&vi.Slow, &vi.Fast, vi.DoGOutTsr.SubSpace(0).(*tensor.Float32)) // on only
		vi.Pos = vi.Pos.Add(vi.Velocity)
	}
	vi.Motion.StarMotion(&vi.MotionOut, &vi.Slow, &vi.Fast)
}

// RenderFrame renders a frame
func (vi *Vis) RenderFrame() {
	tensor.SetAllFloat64(&vi.ImageTsr, 0)
	for y := range vi.Bar.Y {
		yp, _ := edge.Edge(y+vi.Pos.Y, vi.ImageSize.Y, true)
		for x := range vi.Bar.X {
			xp, _ := edge.Edge(x+vi.Pos.X, vi.ImageSize.X, true)
			vi.ImageTsr.Set(1, yp, xp)
		}
	}
}

// LGNDoG runs DoG filtering on input image
// must have valid Image in place to start.
func (vi *Vis) LGNDoG() {
	flt := vi.DoG.FilterTensor(&vi.DoGFilter, dog.Net)
	out := &vi.DoGOutTsr
	vfilter.Conv1(&vi.Geom, flt, &vi.ImageTsr, out, vi.DoG.Gain)
	// log norm is generally good it seems for dogs
	n := out.Len()
	for i := range n {
		out.SetFloat1D(math.Log(out.Float1D(i)+1), i)
	}
	mx := stats.Max(tensor.As1D(out))
	tmath.DivOut(out, mx, out)
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error { //types:add
	vi.LGNDoG()
	return nil
}

func (vi *Vis) ConfigGUI() *core.Body {
	b := core.NewBody("lgn_dog").SetTitle("LGN DoG Filtering")
	core.NewForm(b).SetStruct(vi)
	b.AddTopBar(func(bar *core.Frame) {
		core.NewToolbar(bar).Maker(func(p *tree.Plan) {
			tree.Add(p, func(w *core.FuncButton) { w.SetFunc(vi.RenderFrames) })
		})
	})
	b.RunMainWindow()
	return b
}
