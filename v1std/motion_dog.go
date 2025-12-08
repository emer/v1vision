// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1std

import (
	"image"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/dog"
	"github.com/emer/v1vision/motion"
	"github.com/emer/v1vision/v1vision"
)

// MotionDoG computes starburst-amacrine style motion processing and
// resulting summary full-field motion values, on greyscale
// difference-of-gaussian (DoG) filtering.
// Call Defaults and then set any custom params, then call Config.
// Results are in Output tensor after Run().
type MotionDoG struct {
	// GPU means use the GPU by default (does GPU initialization) in Config.
	// To change what is actually used at the moment of running,
	// set [v1vision.UseGPU].
	GPU bool

	// LGN DoG filter parameters.
	DoG dog.Filter

	// Motion filter parameters.
	Motion motion.Params

	// Geom is geometry of input, output.
	Geom v1vision.Geom `edit:"-"`

	// FullField has the integrated FullField output (1D).
	// Use [motion.Directions] for 1D indexes (is 2x2 for [L,R][D,U]).
	FullField tensor.Float32 `display:"no-inline"`

	// GetStar retrieves the star values. Otherwise, just the full-field.
	GetStar bool

	// V1 is the V1Vision filter processing system.
	V1 v1vision.V1Vision `display:"no-inline"`

	// Star has the star values, if GetStar is true,
	// pointing to Values in V1.
	// [Y, X, Polarity, 4], where Polarity is DoG polarity, and 4 is for
	// Left, Right, Down, Up.
	Star *tensor.Float32 `display:"no-inline"`
}

func (vi *MotionDoG) Defaults() {
	vi.GPU = true
	vi.DoG.Defaults()
	vi.Motion.Defaults()
	vi.SetSize(12, 4)
}

// SetSize sets the V1sGabor filter size and geom spacing to given values.
// Default is 12, 4, for a medium-sized filter.
func (vi *MotionDoG) SetSize(sz, spc int) {
	vi.DoG.Spacing = spc
	vi.DoG.Size = sz
	vi.Geom.Set(math32.Vec2i(0, 0), math32.Vec2i(spc, spc), math32.Vec2i(sz, sz))
}

// Config configures the filtering pipeline with all the current parameters.
// imageSize is the _content_ size of input image that is passed to Run
// as an RGB Tensor (per [V1Vision.Images] standard format),
// (i.e., exclusive of the additional border around the image = [Image.Size]).
// The resulting Geom.Border field can be passed to [Image] methods.
func (vi *MotionDoG) Config(imageSize image.Point) {
	vi.Geom.SetImageSize(imageSize)
	vi.FullField.SetShapeSizes(2, 2)

	fn := 1 // number of filters in DoG

	vi.V1.Init()
	img := vi.V1.NewImage(vi.Geom.In.V())
	wrap := vi.V1.NewImage(vi.Geom.In.V())

	vi.V1.NewWrapImage(img, 0, wrap, int(vi.Geom.Border.X), &vi.Geom)
	_, out := vi.V1.NewDoG(wrap, 0, &vi.DoG, &vi.Geom)
	vi.V1.NewLogValues(out, out, fn, 1.0, &vi.Geom)
	vi.V1.NewNormDiv(v1vision.MaxScalar, out, out, fn, &vi.Geom)

	vi.Motion.DoGSumScalarIndex = vi.V1.NewAggScalar(v1vision.SumScalar, out, fn, &vi.Geom)
	fastIdx := vi.V1.NewMotionIntegrate(out, fn, vi.Motion.FastTau, vi.Motion.SlowTau, &vi.Geom)
	starIdx := vi.V1.NewMotionStar(fastIdx, fn, vi.Motion.Gain, &vi.Geom)
	vi.Motion.FFScalarIndex = vi.V1.NewMotionFullField(starIdx, fn, &vi.Geom)

	if vi.GetStar {
		out4 := vi.V1.NewValues4D(int(vi.Geom.Out.Y), int(vi.Geom.Out.X), 2, 4)
		vi.Star.SetShapeSizes(int(vi.Geom.Out.Y), int(vi.Geom.Out.X), 2, 4)
		vi.V1.NewTo4D(starIdx, out4, 2, 4, 0, &vi.Geom)
	}

	vi.V1.SetAsCurrent()
	if vi.GPU {
		vi.V1.GPUInit()
	}
}

// RunImage runs the configured filtering pipeline
// on given Image, using given [Image] handler.
func (vi *MotionDoG) RunImage(im *Image, img image.Image) {
	im.SetImageGrey(&vi.V1, img, int(vi.Geom.Border.X))
	vi.Run()
}

// Run runs the configured filtering pipeline
// on given Image tensor.
func (vi *MotionDoG) RunTensor(tsr *tensor.Float32) {
	itsr := vi.V1.Images.SubSpace(0).(*tensor.Float32)
	itsr.CopyFrom(tsr)
	vi.Run()
}

// Run runs the configured filtering pipeline.
// image in vi.V1.Images[0] must already have been set.
func (vi *MotionDoG) Run() {
	v1vision.UseGPU = vi.GPU
	vi.V1.SetAsCurrent()

	vals := []v1vision.GPUVars{v1vision.ScalarsVar, v1vision.ValuesVar}
	if vi.GetStar {
		vals = append(vals, v1vision.Values4DVar)
	}
	vi.V1.Run(vals...)
	vi.Motion.FullFieldInteg(vi.V1.Scalars, &vi.FullField)
	if vi.GetStar {
		vi.Star = vi.V1.Values4D.SubSpace(0).(*tensor.Float32)
	}
}

// Init resets all motion integration values to 0.
func (vi *MotionDoG) Init() {
	vi.V1.SetAsCurrent()
	tensor.SetAllFloat64(vi.V1.Values, 0)
	tensor.SetAllFloat64(vi.V1.Scalars, 0)
	tensor.SetAllFloat64(&vi.FullField, 0)
	vi.V1.ToGPUInfra()
}
