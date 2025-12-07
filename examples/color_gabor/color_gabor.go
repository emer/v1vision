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
	_ "cogentcore.org/lab/gosl/slbool/slboolcore" // include to get gui views
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorcore"
	_ "cogentcore.org/lab/tensorcore" // include to get gui views
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/v1vision/gabor"
	"github.com/emer/v1vision/kwta"
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

	// SplitColor records separate rows in V1c simple summary for each color.
	// Otherwise records the max across all colors.
	SplitColor bool

	// ColorGain is an extra gain for color channels, which are lower contrast in general.
	ColorGain float32 `default:"8"`

	// name of image file to operate on
	ImageFile core.Filename

	// V1 simple gabor filter parameters
	V1sGabor gabor.Filter

	// geometry of input, output for V1 simple-cell processing
	V1sGeom v1vision.Geom `edit:"-"`

	// geometry of input, output for V1 complex-cell processing from V1s inputs.
	V1cGeom v1vision.Geom `edit:"-"`

	// neighborhood inhibition for V1s. Each unit gets inhibition from
	// same feature in nearest orthogonal neighbors.
	// Reduces redundancy of feature code.
	V1sNeighInhib kwta.NeighInhib

	// kwta parameters for V1s
	V1sKWTA kwta.KWTA

	// target image size to use -- images will be rescaled to this size
	ImageSize image.Point

	// V1 simple gabor filter tensor
	V1sGaborTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter table (view only)
	V1sGaborTable table.Table `display:"no-inline"`

	// current input image
	Image image.Image `display:"-"`

	// V1 is the V1Vision filter processing system
	V1 v1vision.V1Vision `display:"no-inline"`

	// input image as tensor: original in full color.
	ImageTsr *tensor.Float32 `display:"no-inline"`

	// input image as tensor: visual-system Long, Medium, Short (~R,G,B) filtered
	// with R = grey, G = Red - Green, B = Blue - Yellow opponents.
	ImageLMSTsr *tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, kwta output tensor, Grey = White-Black
	V1sGreyTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, kwta output tensor, Red-Green
	V1sRedGreenTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, kwta output tensor, Blue-Yellow
	V1sBlueYellowTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, kwta output tensor,
	// Max over Grey, RedGreen, BlueYellow
	V1sMaxTsr tensor.Float32 `display:"no-inline"`

	// V1 complex gabor filter output, max-polarity (angle-only) features tensor
	V1cMaxPolTsr tensor.Float32 `display:"no-inline"`

	// V1 complex gabor filter output, max-pooled 2x2 of MaxPol tensor
	V1cPolPoolTsr tensor.Float32 `display:"no-inline"`

	// V1 complex length sum filter output tensor
	V1cLenSumTsr tensor.Float32 `display:"no-inline"`

	// V1 complex end stop filter output tensor
	V1cEndStopTsr tensor.Float32 `display:"no-inline"`

	// Output has the resulting V1c filter outputs, pointing to Values4D in V1.
	// Inner Y, X dimensions are 5 x 4, where the 4 are the gabor angles
	// (0, 45, 90, 135) and the 5 are: 1 length-sum, 2 directions of end-stop,
	// and 2 polarities of V1simple, or 6 with 3 from each LMS opponent channel
	// for SplitColor (9 x 4)
	V1AllTsr *tensor.Float32 `display:"no-inline"`

	// V1cColor is an encapsulated version of this functionality,
	// which we test here for comparison.
	V1cColor v1std.V1cColor

	// StdImage manages images for V1cColor
	StdImage v1std.Image

	// V1 complex gabor filter output, un-max-pooled 2x2 of V1cPool tensor
	V1cUnPoolTsr tensor.Float32 `display:"no-inline"`
	// input image reconstructed from V1s tensor
	ImageFromV1sTsr tensor.Float32 `display:"no-inline"`

	tabView *core.Tabs

	v1sIdxs [3]int

	fadeOpIdx int

	v1sMaxIdx, v1cPoolIdx, v1cMaxPolIdx, v1cPolPoolIdx, v1cLenSumIdx, v1cEndStopIdx int
}

func (vi *Vis) Defaults() {
	vi.GPU = true
	vi.ColorGain = 8
	vi.SplitColor = true
	vi.ImageFile = core.Filename("car_004_00001.png")
	vi.V1sGabor.Defaults()
	sz := 12 // V1mF16 typically = 12, no border
	spc := 4
	vi.V1sGabor.SetSize(sz, spc)
	vi.ImageSize = image.Point{128, 128}
	// vi.ImageSize = image.Point{256, 256}
	// vi.ImageSize = image.Point{512, 512}

	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	vi.V1sGeom.SetImage(math32.Vec2i(0, 0), math32.Vec2i(spc, spc), math32.Vec2i(sz, sz), vi.ImageSize)
	vi.V1sNeighInhib.Defaults()
	vi.V1sKWTA.Defaults()

	vi.V1cColor.Defaults()
	vi.V1cColor.GPU = false
	vi.StdImage.Defaults()
}

// Config sets up the V1 processing pipeline.
func (vi *Vis) Config() {
	vi.V1.Init()
	*vi.V1.NewKWTAParams() = vi.V1sKWTA
	kwtaIdx := 0
	_ = kwtaIdx
	img := vi.V1.NewImage(vi.V1sGeom.In.V())
	wrap := vi.V1.NewImage(vi.V1sGeom.In.V())
	lms := vi.V1.NewImage(vi.V1sGeom.In.V())
	vi.ImageTsr = vi.V1.Images.SubSpace(img).(*tensor.Float32)
	vi.ImageLMSTsr = vi.V1.Images.SubSpace(lms).(*tensor.Float32)

	vi.fadeOpIdx = vi.V1.NewFadeImage(img, 3, wrap, int(vi.V1sGeom.FilterRt.X), .5, .5, .5, &vi.V1sGeom)
	vi.V1.NewLMSOpponents(wrap, lms, vi.ColorGain, &vi.V1sGeom)

	nang := vi.V1sGabor.NAngles

	// V1s simple
	ftyp := vi.V1.NewFilter(nang, vi.V1sGabor.Size, vi.V1sGabor.Size)
	vi.V1.GaborToFilter(ftyp, &vi.V1sGabor)
	inh := vi.V1.NewInhibs(int(vi.V1sGeom.Out.Y), int(vi.V1sGeom.Out.X))
	lmsMap := [3]int{1, int(v1vision.RedGreen), int(v1vision.BlueYellow)}
	for irgb := range 3 {
		out := vi.V1.NewConvolveImage(lms, lmsMap[irgb], ftyp, nang, vi.V1sGabor.Gain, &vi.V1sGeom)
		v1out := out
		if vi.V1sKWTA.On.IsTrue() {
			ninh := 0
			if vi.V1sNeighInhib.On {
				ninh = vi.V1.NewNeighInhib4(out, nang, vi.V1sNeighInhib.Gi, &vi.V1sGeom)
			}
			v1out = vi.V1.NewKWTA(out, ninh, nang, kwtaIdx, inh, &vi.V1sGeom)
		}
		vi.v1sIdxs[irgb] = v1out
	}
	mcout := vi.V1.NewValues(int(vi.V1sGeom.Out.Y), int(vi.V1sGeom.Out.X), nang)
	vi.v1sMaxIdx = mcout
	vi.V1.NewMaxCopy(vi.v1sIdxs[0], vi.v1sIdxs[1], mcout, nang, &vi.V1sGeom)
	vi.V1.NewMaxCopy(vi.v1sIdxs[2], mcout, mcout, nang, &vi.V1sGeom)

	// V1c complex
	vi.V1cGeom.SetFilter(math32.Vec2i(0, 0), math32.Vec2i(2, 2), math32.Vec2i(2, 2), vi.V1sGeom.Out.V())

	mpout := vi.V1.NewMaxPolarity(mcout, nang, &vi.V1sGeom)
	vi.v1cMaxPolIdx = mpout
	pmpout := vi.V1.NewMaxPool(mpout, 1, nang, &vi.V1cGeom)
	vi.v1cPolPoolIdx = pmpout
	lsout := vi.V1.NewLenSum4(pmpout, nang, &vi.V1cGeom)
	vi.v1cLenSumIdx = lsout
	esout := vi.V1.NewEndStop4(pmpout, lsout, nang, &vi.V1cGeom)
	vi.v1cEndStopIdx = esout

	// To4D
	out4Rows := 5
	if vi.SplitColor {
		out4Rows = 9
	}
	out4 := vi.V1.NewValues4D(int(vi.V1cGeom.Out.Y), int(vi.V1cGeom.Out.X), out4Rows, nang)
	vi.V1.NewTo4D(lsout, out4, 1, nang, 0, &vi.V1cGeom)
	vi.V1.NewTo4D(esout, out4, 2, nang, 1, &vi.V1cGeom)
	if vi.SplitColor {
		poutg := vi.V1.NewMaxPool(vi.v1sIdxs[0], 2, nang, &vi.V1cGeom)
		poutrg := vi.V1.NewMaxPool(vi.v1sIdxs[1], 2, nang, &vi.V1cGeom)
		poutby := vi.V1.NewMaxPool(vi.v1sIdxs[2], 2, nang, &vi.V1cGeom)
		vi.V1.NewTo4D(poutg, out4, 2, nang, 3, &vi.V1cGeom)
		vi.V1.NewTo4D(poutrg, out4, 2, nang, 5, &vi.V1cGeom)
		vi.V1.NewTo4D(poutby, out4, 2, nang, 7, &vi.V1cGeom)
	} else {
		pout := vi.V1.NewMaxPool(mcout, 2, nang, &vi.V1cGeom)
		vi.V1.NewTo4D(pout, out4, 2, nang, 3, &vi.V1cGeom)
	}

	vi.V1.SetAsCurrent()
	if vi.GPU {
		vi.V1.GPUInit()
	}

	vi.V1cColor.Config(vi.StdImage.Size)
}

func (vi *Vis) getTsr(idx int, tsr *tensor.Float32, y, x, pol int32) {
	out := vi.V1.Values.SubSpace(idx).(*tensor.Float32)
	tsr.SetShapeSizes(int(y), int(x), int(pol), vi.V1sGabor.NAngles)
	tensor.CopyFromLargerShape(tsr, out)
}

func (vi *Vis) getTsrOpt(idx int, tsr *tensor.Float32, y, x, pol int32) {
	if idx == 0 {
		return
	}
	vi.getTsr(idx, tsr, y, x, pol)
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error { //types:add
	// key point here: it is not re-sending the background guys
	// so the other ones from v1c are interfering.
	// need a set as current that also uploads backgrounds

	vi.V1.SetAsCurrent()
	v1vision.UseGPU = vi.GPU
	err := vi.OpenImage(string(vi.ImageFile))
	if err != nil {
		return errors.Log(err)
	}
	r, g, b := v1vision.EdgeAvg(vi.ImageTsr, int(vi.V1sGeom.FilterRt.X))
	vi.V1.SetFadeRGB(vi.fadeOpIdx, r, g, b)

	tmr := timer.Time{}
	tmr.Start()
	for range 1000 {
		vi.V1.Run()
		// vi.V1.Run(v1vision.Values4DVar) // this is sig slower due to sync issues.
		// for timing test, run without sync and assume it gets sig better.
	}
	tmr.Stop()
	fmt.Println("GPU:", vi.GPU, "Time:", tmr.Total)
	// With 10 Iters on KWTA, on MacBookPro M3Pro:
	// 128 image: CPU: 6.3s, GPU: 1.67s
	// 256 image: CPU: 15.5s, GPU: 913ms
	// 512 image: CPU: 49.2s, GPU: 3.5s (7.9s with Values4D sync)
	// note: not sending image at start is the same!

	vi.V1.Run(v1vision.Values4DVar, v1vision.ValuesVar, v1vision.ImagesVar)

	vi.getTsr(vi.v1sIdxs[0], &vi.V1sGreyTsr, vi.V1sGeom.Out.Y, vi.V1sGeom.Out.X, 2)
	vi.getTsrOpt(vi.v1sIdxs[1], &vi.V1sRedGreenTsr, vi.V1sGeom.Out.Y, vi.V1sGeom.Out.X, 2)
	vi.getTsrOpt(vi.v1sIdxs[2], &vi.V1sBlueYellowTsr, vi.V1sGeom.Out.Y, vi.V1sGeom.Out.X, 2)
	vi.getTsrOpt(vi.v1sMaxIdx, &vi.V1sMaxTsr, vi.V1sGeom.Out.Y, vi.V1sGeom.Out.X, 2)

	vi.getTsrOpt(vi.v1cMaxPolIdx, &vi.V1cMaxPolTsr, vi.V1sGeom.Out.Y, vi.V1sGeom.Out.X, 1)
	vi.getTsrOpt(vi.v1cPolPoolIdx, &vi.V1cPolPoolTsr, vi.V1cGeom.Out.Y, vi.V1cGeom.Out.X, 1)
	vi.getTsrOpt(vi.v1cLenSumIdx, &vi.V1cLenSumTsr, vi.V1cGeom.Out.Y, vi.V1cGeom.Out.X, 1)
	vi.getTsrOpt(vi.v1cEndStopIdx, &vi.V1cEndStopTsr, vi.V1cGeom.Out.Y, vi.V1cGeom.Out.X, 2)

	vi.V1AllTsr = vi.V1.Values4D.SubSpace(0).(*tensor.Float32)

	// vi.ImageFromV1Simple()

	vi.V1cColor.RunImage(&vi.StdImage, vi.Image)

	if vi.tabView != nil {
		vi.tabView.Update()
	}

	return nil
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
	v1vision.RGBToTensor(vi.Image, vi.ImageTsr, int(vi.V1sGeom.FilterRt.X), v1vision.BottomZero)
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

func (vi *Vis) ConfigGUI() *core.Body {
	vi.V1sGaborTable.Init()
	vi.V1sGabor.ToTable(&vi.V1sGaborTable) // note: view only, testing
	tensorcore.AddGridStylerTo(vi.ImageTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(vi.ImageLMSTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(&vi.ImageFromV1sTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(&vi.V1sGaborTable, func(s *tensorcore.GridStyle) {
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
	tf, _ = tb.NewTab("Image LMS")
	tensorcore.NewTensorGrid(tf).SetTensor(vi.ImageLMSTsr)
	tf, _ = tb.NewTab("V1All")
	tensorcore.NewTensorGrid(tf).SetTensor(vi.V1AllTsr)

	tf, _ = tb.NewTab("V1s Grey")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.V1sGreyTsr)
	tf, _ = tb.NewTab("V1s Red - Green")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.V1sRedGreenTsr)
	tf, _ = tb.NewTab("V1s Blue - Yellow")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.V1sBlueYellowTsr)
	tf, _ = tb.NewTab("V1s Max")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.V1sMaxTsr)

	tf, _ = tb.NewTab("V1cMaxPol")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.V1cMaxPolTsr)
	tf, _ = tb.NewTab("V1cPolPool")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.V1cPolPoolTsr)
	tf, _ = tb.NewTab("V1cLenSum")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.V1cLenSumTsr)
	tf, _ = tb.NewTab("V1cEndStop")
	tensorcore.NewTensorGrid(tf).SetTensor(&vi.V1cEndStopTsr)

	sp.SetSplits(.3, .7)

	b.AddTopBar(func(bar *core.Frame) {
		core.NewToolbar(bar).Maker(func(p *tree.Plan) {
			tree.Add(p, func(w *core.FuncButton) { w.SetFunc(vi.Filter) })
		})
	})
	b.RunMainWindow()
	return b
}
