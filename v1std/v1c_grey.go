// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1std

import (
	"image"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/gabor"
	"github.com/emer/v1vision/kwta"
	"github.com/emer/v1vision/v1vision"
)

// V1cGrey does greyscale V1 complex (V1c) filtering, starting with
// simple cells (V1s) and adding length sum and end stopping.
// KWTA inhibition operates on the V1s step.
// Call Defaults and then set any custom params, then call Config.
// Results are in Output tensor after Run(), which has a 4D shape.
type V1cGrey struct {
	// GPU means use the GPU by default (does GPU initialization) in Config.
	// To change what is actually used at the moment of running,
	// set [v1vision.UseGPU].
	GPU bool

	// V1 simple gabor filter parameters
	V1sGabor gabor.Filter

	// V1sNeighInhib specifies neighborhood inhibition for V1s.
	// Each unit gets inhibition from same feature in nearest orthogonal
	// neighbors. Reduces redundancy of feature code.
	V1sNeighInhib kwta.NeighInhib

	// V1sKWTA has the kwta inhibition parameters for V1s.
	V1sKWTA kwta.KWTA

	// geometry of input, output for V1 simple-cell processing.
	V1sGeom v1vision.Geom `edit:"-"`

	// geometry of input, output for V1 complex-cell processing from V1s inputs.
	V1cGeom v1vision.Geom `edit:"-"`

	// V1 is the V1Vision filter processing system
	V1 v1vision.V1Vision `display:"no-inline"`

	// Output has the resulting V1c filter outputs, pointing to Values4D in V1.
	// Inner Y, X dimensions are 5 x 4, where the 4 are the gabor angles
	// (0, 45, 90, 135) and the 5 are: 1 length-sum, 2 directions of end-stop,
	// and 2 polarities of V1simple.
	Output *tensor.Float32 `display:"no-inline"`
}

func (vi *V1cGrey) Defaults() {
	vi.GPU = true
	vi.V1sGabor.Defaults()
	vi.V1sNeighInhib.Defaults()
	vi.V1sKWTA.Defaults()
	vi.SetSize(12, 4)
}

// SetSize sets the V1sGabor filter size and geom spacing to given values.
// Default is 12, 4, for a medium-sized filter.
func (vi *V1cGrey) SetSize(sz, spc int) {
	vi.V1sGabor.SetSize(sz, spc)
	vi.V1sGeom.Set(math32.Vec2i(0, 0), math32.Vec2i(spc, spc), math32.Vec2i(sz, sz))
}

// Config configures the filtering pipeline with all the current parameters.
// imageSize is the _content_ size of input image that is passed to Run
// as an RGB Tensor (per [V1Vision.Images] standard format),
// (i.e., exclusive of the additional border around the image = [Image.Size]).
// The resulting Geom.Border field can be passed to [Image] methods.
func (vi *V1cGrey) Config(imageSize image.Point) {
	vi.V1sGeom.SetImageSize(imageSize)

	vi.V1.Init()
	*vi.V1.NewKWTAParams() = vi.V1sKWTA
	kwtaIdx := 0
	img := vi.V1.NewImage(vi.V1sGeom.In.V())
	wrap := vi.V1.NewImage(vi.V1sGeom.In.V())

	vi.V1.NewWrapImage(img, 0, wrap, int(vi.V1sGeom.FilterRt.X), &vi.V1sGeom)

	nang := vi.V1sGabor.NAngles

	// V1s simple
	_, out := vi.V1.NewGabor(wrap, 0, &vi.V1sGabor, &vi.V1sGeom)
	v1out := out
	if vi.V1sKWTA.On.IsTrue() {
		ninh := 0
		if vi.V1sNeighInhib.On {
			ninh = vi.V1.NewNeighInhib4(out, nang, vi.V1sNeighInhib.Gi, &vi.V1sGeom)
		}
		inh := vi.V1.NewInhibs(int(vi.V1sGeom.Out.Y), int(vi.V1sGeom.Out.X))
		v1out = vi.V1.NewKWTA(out, ninh, nang, kwtaIdx, inh, &vi.V1sGeom)
	}

	// V1c complex
	vi.V1cGeom.SetFilter(math32.Vec2i(0, 0), math32.Vec2i(2, 2), math32.Vec2i(2, 2), vi.V1sGeom.Out.V())
	pout := vi.V1.NewMaxPool(v1out, 2, nang, &vi.V1cGeom)
	mpout := vi.V1.NewMaxPolarity(v1out, nang, &vi.V1sGeom)
	pmpout := vi.V1.NewMaxPool(mpout, 1, nang, &vi.V1cGeom)
	lsout := vi.V1.NewLenSum4(pmpout, nang, &vi.V1cGeom)
	esout := vi.V1.NewEndStop4(pmpout, lsout, nang, &vi.V1cGeom)

	// To4D
	out4 := vi.V1.NewValues4D(int(vi.V1cGeom.Out.Y), int(vi.V1cGeom.Out.X), 5, nang)
	vi.V1.NewTo4D(lsout, out4, 1, nang, 0, &vi.V1cGeom)
	vi.V1.NewTo4D(esout, out4, 2, nang, 1, &vi.V1cGeom)
	vi.V1.NewTo4D(pout, out4, 2, nang, 3, &vi.V1cGeom)

	vi.V1.SetAsCurrent()
	if vi.GPU {
		vi.V1.GPUInit()
	}
}

// RunImage runs the configured filtering pipeline.
// on given Image, using given [Image] handler.
func (vi *V1cGrey) RunImage(im *Image, img image.Image) {
	vi.V1.SetAsCurrent()
	v1vision.UseGPU = vi.GPU
	im.SetImageGrey(&vi.V1, img, int(vi.V1sGeom.Border.X))
	vi.V1.Run(v1vision.Values4DVar)
	vi.Output = vi.V1.Values4D.SubSpace(0).(*tensor.Float32)
}
