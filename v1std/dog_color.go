// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1std

import (
	"image"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/dog"
	"github.com/emer/v1vision/kwta"
	"github.com/emer/v1vision/v1vision"
)

// DoGColor does color difference-of-gaussian (DoG) filtering,
// on Red - Green and Blue - Yellow opponent color contrasts,
// so that activity reflects presence of a color beyond grey baseline.
// These capture the activity of the blob chroma sensitive cells.
// Call Defaults and then set any custom params, then call Config.
// Results are in Output tensor after Run().
type DoGColor struct {
	// GPU means use the GPU by default (does GPU initialization) in Config.
	// To change what is actually used at the moment of running,
	// set [v1vision.UseGPU].
	GPU bool

	// LGN DoG filter parameters. Generally have larger fields,
	// and no spatial tuning (i.e., OnSigma == OffSigma), consistent
	// with blob cells.
	DoG dog.Filter

	// Geom is geometry of input, output.
	Geom v1vision.Geom `edit:"-"`

	// kwta parameters, providing more contrast across colors.
	KWTA kwta.KWTA

	// V1 is the V1Vision filter processing system.
	V1 v1vision.V1Vision `display:"no-inline"`

	// Output has the resulting DoG filter outputs, pointing to Values in V1.
	// [Y, X, Polarity, Feature], where Polarity = On (0) vs Off (1) stronger.
	// Feature: 0 = Red vs. Green; 1 = Blue vs. Yellow.
	Output *tensor.Float32 `display:"no-inline"`

	outIdx int
}

func (vi *DoGColor) Defaults() {
	vi.GPU = true
	vi.DoG.Defaults()
	vi.DoG.Gain = 8 // color channels are weaker than grey
	vi.DoG.OnGain = 1
	vi.DoG.SetSameSigma(0.5) // no spatial component, just pure contrast
	vi.SetSize(12, 16)       // V1mF16 typically = 12, no border
	vi.KWTA.Defaults()
	vi.KWTA.Layer.On.SetBool(false) // non-spatial, mainly for differentiation within pools
	vi.KWTA.Pool.Gi = 1.2
}

// SetSize sets the V1sGabor filter size and geom spacing to given values.
// Default is 12, 16, for a medium-sized filter.
func (vi *DoGColor) SetSize(sz, spc int) {
	vi.DoG.Spacing = spc
	vi.DoG.Size = sz
	vi.Geom.Set(math32.Vec2i(0, 0), math32.Vec2i(spc, spc), math32.Vec2i(sz, sz))
}

// Config configures the filtering pipeline with all the current parameters.
// imageSize is the _content_ size of input image that is passed to Run
// as an RGB Tensor (per [V1Vision.Images] standard format),
// (i.e., exclusive of the additional border around the image = [Image.Size]).
// The resulting Geom.Border field can be passed to [Image] methods.
func (vi *DoGColor) Config(imageSize image.Point) {
	vi.Geom.SetImageSize(imageSize)

	vi.V1.Init()
	*vi.V1.NewKWTAParams() = vi.KWTA
	kwtaIdx := 0
	img := vi.V1.NewImage(vi.Geom.In.V())
	wrap := vi.V1.NewImage(vi.Geom.In.V())
	lmsRG := vi.V1.NewImage(vi.Geom.In.V())
	lmsBY := vi.V1.NewImage(vi.Geom.In.V())

	vi.V1.NewWrapImage(img, 3, wrap, int(vi.Geom.Border.X), &vi.Geom)
	vi.V1.NewLMSComponents(wrap, lmsRG, lmsBY, vi.DoG.Gain, &vi.Geom)

	out := vi.V1.NewValues(int(vi.Geom.Out.Y), int(vi.Geom.Out.X), 2)
	dogFt := vi.V1.NewDoGOnOff(&vi.DoG, &vi.Geom)

	vi.V1.NewConvolveDiff(lmsRG, v1vision.Red, lmsRG, v1vision.Green, dogFt, 0, 1, out, 0, 1, vi.DoG.OnGain, &vi.Geom)
	vi.V1.NewConvolveDiff(lmsBY, v1vision.Blue, lmsBY, v1vision.Yellow, dogFt, 0, 1, out, 1, 1, vi.DoG.OnGain, &vi.Geom)

	vi.outIdx = out
	if vi.KWTA.On.IsTrue() {
		inh := vi.V1.NewInhibs(int(vi.Geom.Out.Y), int(vi.Geom.Out.X))
		vi.outIdx = vi.V1.NewKWTA(out, 0, 2, kwtaIdx, inh, &vi.Geom)
	}

	vi.V1.SetAsCurrent()
	if vi.GPU {
		vi.V1.GPUInit()
	}
}

// RunImage runs the configured filtering pipeline.
// on given Image, using given [Image] handler.
func (vi *DoGColor) RunImage(im *Image, img image.Image) {
	v1vision.UseGPU = vi.GPU
	vi.V1.SetAsCurrent()
	im.SetImageRGB(&vi.V1, img, int(vi.Geom.Border.X))
	vi.V1.Run(v1vision.ValuesVar)
	vi.Output = vi.V1.Values.SubSpace(vi.outIdx).(*tensor.Float32)
}
