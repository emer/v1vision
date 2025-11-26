// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate core generate -add-types

import (
	"fmt"
	"math"
	"time"

	"cogentcore.org/core/core"
	"cogentcore.org/core/events"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
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
	"github.com/emer/v1vision/v1vision"
)

func main() {
	vi := &Vis{}
	vi.Defaults()
	vi.RenderFrames()
	vi.ConfigGUI()
}

// Vis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed
type Vis struct { //types:add

	// NFrames is the number of frames to render per trial.
	NFrames int

	// FrameDelay is time per frame for display updating.
	FrameDelay time.Duration

	// target image size to use.
	ImageSize math32.Vector2i

	// Bar is the size of the moving bar.
	Bar math32.Vector2i

	// Velocity is the motion direction vector.
	Velocity math32.Vector2

	// Start is the starting position.
	Start math32.Vector2

	// Pos is current position
	Pos math32.Vector2 `edit:"-"`

	// Motion filter parameters.
	Motion motion.Params

	// LGN DoG filter parameters
	DoG dog.Filter

	// geometry of input, output
	Geom v1vision.Geom `edit:"-"`

	// input image as tensor
	ImageTsr tensor.Float32 `display:"no-inline"`

	// DoG filter tensor -- has 3 filters (on, off, net)
	DoGFilter tensor.Float32 `display:"no-inline"`

	// DoG filter table (view only)
	DoGTab table.Table `display:"no-inline"`

	// DoG filter output tensor
	DoGOutTsr tensor.Float32 `display:"no-inline"`

	Slow           tensor.Float32 `display:"no-inline"`
	Fast           tensor.Float32 `display:"no-inline"`
	Star           tensor.Float32 `display:"no-inline"`
	FullField      tensor.Float32 `display:"no-inline"`
	FullFieldInsta tensor.Float32 `display:"no-inline"`

	starView, fastView, imgView *tensorcore.TensorGrid
}

func (vi *Vis) Defaults() {
	vi.NFrames = 16
	vi.FrameDelay = 200 * time.Millisecond
	vi.ImageSize = math32.Vector2i{64, 64}
	vi.ImageTsr.SetShapeSizes(64, 64)
	vi.Bar = math32.Vector2i{8, 16}
	vi.Velocity = math32.Vector2{1, 0}
	vi.Start = math32.Vector2{8, 8}
	vi.DoGTab.Init()
	vi.Motion.Defaults()
	vi.DoG.Defaults()
	sz := 12 // V1mF16 typically = 12, no border
	spc := 4
	vi.DoG.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.Geom.Set(math32.Vector2i{0, 0}, math32.Vector2i{spc, spc}, math32.Vector2i{sz, sz})
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
	tensor.SetAllFloat64(&vi.Slow, 0)
	tensor.SetAllFloat64(&vi.Fast, 0)
	vi.Pos = vi.Start
	visNorm := float32(0)
	for i := range vi.NFrames {
		_ = i
		vi.RenderFrame()
		vi.LGNDoG()
		ve := vi.Motion.IntegrateFrame(&vi.Slow, &vi.Fast, &vi.DoGOutTsr)
		vi.Pos = vi.Pos.Add(vi.Velocity)
		vi.Motion.StarMotion(&vi.Star, &vi.Slow, &vi.Fast)
		vi.Motion.FullField(&vi.FullFieldInsta, &vi.FullField, &vi.Star, ve, &visNorm)
		if vi.starView != nil {
			// fmt.Println(i)
			vi.starView.AsyncLock()
			vi.starView.Update()
			vi.fastView.Update()
			vi.imgView.Update()
			vi.starView.AsyncUnlock()
			fmt.Printf("%d\tL: %7.4g\tR: %7.4g\tB: %7.4g\tT: %7.4g\tN: %7.4g\n", i, vi.FullField.Value1D(0), vi.FullField.Value1D(1), vi.FullField.Value1D(2), vi.FullField.Value1D(3), visNorm)
			time.Sleep(vi.FrameDelay)
		}
	}
}

// RenderFrame renders a frame
func (vi *Vis) RenderFrame() {
	tensor.SetAllFloat64(&vi.ImageTsr, 0)
	for y := range vi.Bar.Y {
		py := int(math32.Round(vi.Pos.Y))
		yp, _ := edge.Edge(y+py, vi.ImageSize.Y, true)
		for x := range vi.Bar.X {
			px := int(math32.Round(vi.Pos.X))
			xp, _ := edge.Edge(x+px, vi.ImageSize.X, true)
			vi.ImageTsr.Set(1, yp, xp)
		}
	}
}

// LGNDoG runs DoG filtering on input image
// must have valid Image in place to start.
func (vi *Vis) LGNDoG() {
	flt := vi.DoG.FilterTensor(&vi.DoGFilter, dog.Net)
	out := &vi.DoGOutTsr
	v1vision.Conv1(&vi.Geom, flt, &vi.ImageTsr, out, vi.DoG.Gain)
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
	sp := core.NewSplits(b)
	core.NewForm(sp).SetStruct(vi)
	tb := core.NewTabs(sp)
	tf, _ := tb.NewTab("Star")
	vi.starView = tensorcore.NewTensorGrid(tf).SetTensor(&vi.Star)
	tf, _ = tb.NewTab("Full Field")
	vi.imgView = tensorcore.NewTensorGrid(tf).SetTensor(&vi.FullField)
	tf, _ = tb.NewTab("Fast")
	vi.fastView = tensorcore.NewTensorGrid(tf).SetTensor(&vi.Fast)
	tf, _ = tb.NewTab("Image")
	vi.imgView = tensorcore.NewTensorGrid(tf).SetTensor(&vi.ImageTsr)

	sp.SetSplits(.3, .7)

	b.AddTopBar(func(bar *core.Frame) {
		core.NewToolbar(bar).Maker(func(p *tree.Plan) {
			tree.Add(p, func(w *core.Button) {
				w.SetText("Run").SetIcon(icons.PlayArrow)
				w.OnClick(func(e events.Event) {
					go vi.RenderFrames()
				})
			})
		})
	})
	b.RunMainWindow()
	return b
}
