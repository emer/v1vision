// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate core generate -add-types

import (
	"fmt"
	"image"
	"time"

	"cogentcore.org/core/core"
	"cogentcore.org/core/events"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
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
	vi.Config()
	vi.RenderFrames()
	vi.ConfigGUI()
}

// Vis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed
type Vis struct { //types:add
	// GPU means use gpu
	GPU bool

	// NFrames is the number of frames to render per trial.
	NFrames int

	// FrameDelay is time per frame for display updating.
	FrameDelay time.Duration

	// target image size to use.
	ImageSize image.Point

	// Bar is the size of the moving bar.
	Bar image.Point

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

	// V1 is the V1Vision filter processing system
	V1 v1vision.V1Vision `display:"no-inline"`

	// input image as tensor
	ImageTsr *tensor.Float32 `display:"no-inline"`

	// DoG filter table (view only)
	DoGTab table.Table `display:"no-inline"`

	// DoG filter output tensor
	DoGOut tensor.Float32 `display:"no-inline"`

	// Fast motion integration tensor
	Fast tensor.Float32 `display:"no-inline"`

	// Slow motion integration tensor
	Slow tensor.Float32 `display:"no-inline"`

	// Star motion output tensor
	Star tensor.Float32 `display:"no-inline"`

	// FullField integrated output
	FullField tensor.Float32 `display:"no-inline"`

	fastIdx, starIdx int

	tabView *core.Tabs
}

func (vi *Vis) Defaults() {
	vi.GPU = true
	vi.NFrames = 16
	vi.FrameDelay = 200 * time.Millisecond
	vi.ImageSize = image.Point{64, 64}
	vi.Bar = image.Point{8, 16}
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
	vi.Geom.Set(math32.Vec2i(0, 0), math32.Vec2i(spc, spc), math32.Vec2i(sz, sz))
	vi.Geom.SetImageSize(vi.ImageSize)
}

func (vi *Vis) Config() {
	fn := 1 // number of filters in DoG
	_ = fn
	vi.V1.Init()
	img := vi.V1.NewImage(vi.Geom.In.V())
	vi.ImageTsr = vi.V1.Images.SubSpace(0).(*tensor.Float32)
	_, out := vi.V1.AddDoG(img, &vi.DoG, &vi.Geom)
	vi.V1.NewLogValues(out, out, fn, 1.0, &vi.Geom)
	vi.V1.NewNormDiv(v1vision.MaxScalar, out, out, fn, &vi.Geom)

	vi.Motion.DoGSumScalarIndex = vi.V1.NewAggScalar(v1vision.SumScalar, out, fn, &vi.Geom)
	vi.fastIdx = vi.V1.NewMotionIntegrate(out, fn, vi.Motion.FastTau, vi.Motion.SlowTau, &vi.Geom)
	vi.starIdx = vi.V1.NewMotionStar(vi.fastIdx, fn, vi.Motion.Gain, &vi.Geom)
	vi.Motion.FFScalarIndex = vi.V1.NewMotionFullField(vi.starIdx, fn, &vi.Geom)

	vi.V1.SetAsCurrent()
	if vi.GPU {
		vi.V1.GPUInit()
	}
}

// RenderFrames renders the frames
func (vi *Vis) RenderFrames() { //types:add
	vi.V1.ZeroValues()
	vi.Motion.NormInteg = 0
	vi.Pos = vi.Start
	for i := range vi.NFrames {
		_ = i
		vi.RenderFrame()
		vi.Pos = vi.Pos.Add(vi.Velocity)
		vi.Filter()
		if vi.tabView != nil {
			// fmt.Println(i)
			vi.tabView.AsyncLock()
			vi.tabView.Update()
			// vi.fastView.Update()
			// vi.slowView.Update()
			// vi.imgView.Update()
			vi.tabView.AsyncUnlock()
			fmt.Printf("%d\tL: %7.4g\tR: %7.4g\tB: %7.4g\tT: %7.4g\tN: %7.4g\n", i, vi.FullField.Value1D(0), vi.FullField.Value1D(1), vi.FullField.Value1D(2), vi.FullField.Value1D(3), vi.Motion.NormInteg)
			time.Sleep(vi.FrameDelay)
		}
	}
}

// RenderFrame renders a frame
func (vi *Vis) RenderFrame() {
	pad := vi.Geom.Border.V()
	tensor.SetAllFloat64(vi.ImageTsr, 0)
	for y := range vi.Bar.Y {
		py := int(math32.Round(vi.Pos.Y))
		yp, _ := edge.Edge(y+py, vi.ImageSize.Y, true)
		for x := range vi.Bar.X {
			px := int(math32.Round(vi.Pos.X))
			xp, _ := edge.Edge(x+px, vi.ImageSize.X, true)
			vi.ImageTsr.Set(1, 0, int(pad.Y)+yp, int(pad.X)+xp)
		}
	}
}

// Filter runs the filters on current image.
func (vi *Vis) Filter() error { //types:add
	v1vision.UseGPU = vi.GPU
	vi.V1.Run(v1vision.ScalarsVar, v1vision.ValuesVar, v1vision.ImagesVar)
	// vi.V1.Run(v1vision.ScalarsVar) // minimal fastest case
	vi.Motion.FullFieldInteg(vi.V1.Scalars, &vi.FullField)

	out := vi.V1.Values.SubSpace(0).(*tensor.Float32)
	vi.DoGOut.SetShapeSizes(int(vi.Geom.Out.Y), int(vi.Geom.Out.X), 2, 1)
	tensor.CopyFromLargerShape(&vi.DoGOut, out)

	fast := vi.V1.Values.SubSpace(vi.fastIdx).(*tensor.Float32)
	vi.Fast.SetShapeSizes(int(vi.Geom.Out.Y), int(vi.Geom.Out.X), 2, 1)
	tensor.CopyFromLargerShape(&vi.Fast, fast)

	slow := vi.V1.Values.SubSpace(vi.fastIdx + 1).(*tensor.Float32)
	vi.Slow.SetShapeSizes(int(vi.Geom.Out.Y), int(vi.Geom.Out.X), 2, 1)
	tensor.CopyFromLargerShape(&vi.Slow, slow)

	star := vi.V1.Values.SubSpace(vi.starIdx).(*tensor.Float32)
	vi.Star.SetShapeSizes(int(vi.Geom.Out.Y-1), int(vi.Geom.Out.X-1), 2, 4)
	tensor.CopyFromLargerShape(&vi.Star, star)

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
	tensorcore.AddGridStylerTo(&vi.FullField, func(s *tensorcore.GridStyle) {
		s.Range.SetMin(0)
		s.Range.FixMax = false
	})

	b := core.NewBody("lgn_dog").SetTitle("LGN DoG Filtering")
	sp := core.NewSplits(b)
	core.NewForm(sp).SetStruct(vi)
	tb := core.NewTabs(sp)
	vi.tabView = tb
	tf, _ := tb.NewTab("Star")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.Star)
	tf, _ = tb.NewTab("Full Field")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.FullField)
	tf, _ = tb.NewTab("Fast")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.Fast)
	tf, _ = tb.NewTab("Slow")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.Slow)
	tf, _ = tb.NewTab("DoG")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.DoGOut)
	tf, _ = tb.NewTab("Image")
	tensorcore.NewTensorGrid(tf).SetTensor(vi.ImageTsr)

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
