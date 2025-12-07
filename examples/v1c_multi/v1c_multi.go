// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate core generate -add-types

import (
	"fmt"

	"cogentcore.org/core/base/timer"
	"cogentcore.org/core/core"
	"cogentcore.org/core/tree"
	_ "cogentcore.org/lab/gosl/slbool/slboolcore" // include to get gui views
	"cogentcore.org/lab/tensorcore"
	_ "cogentcore.org/lab/tensorcore" // include to get gui views
	"github.com/emer/v1vision/v1std"
)

func main() {
	vi := &Vis{}
	vi.Defaults()
	vi.Config()
	vi.Filter()
	vi.ConfigGUI()
}

type Vis struct { //types:add
	v1std.V1cMulti

	// ImageFile is the name of image file to operate on.
	ImageFile core.Filename

	tabView *core.Tabs
}

func (vi *Vis) Defaults() {
	vi.V1cMulti.Defaults()
	// vi.GPU = false
	vi.StdLowMed16DegZoom1()
	vi.ImageFile = core.Filename("car_004_00001.png")
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error { //types:add
	err := vi.Image.OpenImageResize(string(vi.ImageFile))
	if err != nil {
		return err
	}
	tmr := timer.Time{}
	tmr.Start()
	for range 1 {
		vi.RunImage(vi.Image.Image)
	}
	tmr.Stop()
	fmt.Println("GPU:", vi.GPU, "Time:", tmr.Total)
	// 100: GPU: 1.17s, CPU: 2s (full transfers)

	if vi.tabView != nil {
		vi.tabView.Update()
	}

	return nil
}

func (vi *Vis) ConfigGUI() *core.Body {
	tensorcore.AddGridStylerTo(vi.Image.Tsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	b := core.NewBody("v1gabor").SetTitle("V1 Gabor Filtering")
	sp := core.NewSplits(b)
	core.NewForm(sp).SetStruct(vi)
	tb := core.NewTabs(sp)
	vi.tabView = tb
	tf, _ := tb.NewTab("Image")
	tensorcore.NewTensorGrid(tf).SetTensor(vi.Image.Tsr)
	for _, vp := range vi.V1cParams {
		tf, _ = tb.NewTab("V1c " + vp.Name)
		tensorcore.NewTensorGrid(tf).SetTensor(&vp.Output)
	}
	for _, vp := range vi.DoGParams {
		tf, _ = tb.NewTab("DoG " + vp.Name)
		tensorcore.NewTensorGrid(tf).SetTensor(&vp.Output)
	}

	sp.SetSplits(.3, .7)

	b.AddTopBar(func(bar *core.Frame) {
		core.NewToolbar(bar).Maker(func(p *tree.Plan) {
			tree.Add(p, func(w *core.FuncButton) { w.SetFunc(vi.Filter) })
		})
	})
	b.RunMainWindow()
	return b
}
