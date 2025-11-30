// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1std

import (
	"image"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/dog"
	"github.com/emer/v1vision/v1vision"
)

// DoGGrey does greyscale difference-of-gaussian (DoG) filtering.
// Output is log-max-normalized.
// Call Defaults and then set any custom params, then call Config.
// Results are in Output tensor after Run().
type DoGGrey struct {
	// GPU means use the GPU by default (does GPU initialization) in Config.
	// To change what is actually used at the moment of running,
	// set [v1vision.UseGPU].
	GPU bool

	// LGN DoG filter parameters.
	DoG dog.Filter

	// Geom is geometry of input, output.
	Geom v1vision.Geom `edit:"-"`

	// V1 is the V1Vision filter processing system
	V1 v1vision.V1Vision `display:"no-inline"`

	// Output has the resulting DoG filter outputs, pointing to Values in V1
	Output *tensor.Float32 `display:"no-inline"`
}

func (vi *DoGGrey) Defaults() {
	vi.GPU = true
	vi.DoG.Defaults()
	spc := 4
	sz := 12
	vi.DoG.Spacing = spc
	vi.DoG.Size = sz
	vi.Geom.Set(math32.Vec2i(0, 0), math32.Vec2i(spc, spc), math32.Vec2i(sz, sz))
}

// Config configures the filtering pipeline with all the current parameters.
// imageSize is the _content_ size of input image that is passed to Run
// as an RGB Tensor (per [V1Vision.Images] standard format),
// (i.e., exclusive of the additional border around the image = [Image.Size]).
// The resulting Geom.Border field can be passed to [Image] methods.
func (vi *DoGGrey) Config(imageSize image.Point) {
	spc := vi.DoG.Spacing
	sz := vi.DoG.Size
	vi.Geom.SetImage(math32.Vec2i(0, 0), math32.Vec2i(spc, spc), math32.Vec2i(sz, sz), imageSize)

	vi.V1.Init()
	img := vi.V1.NewImage(vi.Geom.In.V())
	wrap := vi.V1.NewImage(vi.Geom.In.V())

	vi.V1.NewWrapImage(img, 0, wrap, int(vi.Geom.FilterRt.X), &vi.Geom)
	_, out := vi.V1.AddDoG(wrap, &vi.DoG, &vi.Geom)
	// _ = out
	vi.V1.NewLogValues(out, out, 1, 1.0, &vi.Geom)
	vi.V1.NewNormDiv(v1vision.MaxScalar, out, out, 1, &vi.Geom)
	vi.V1.SetAsCurrent()
	vi.V1.GPUInit()
}

// Run runs the configured filtering pipeline.
// MUST have set the input image as the first [V1Vision.Images],
// e.g., by calling [v1vision.RGBToGrey], via [Image.SetImageGrey].
func (vi *DoGGrey) Run() {
	vi.V1.SetAsCurrent()
	vi.V1.Run(v1vision.ValuesVar)
	vi.Output = vi.V1.Values.SubSpace(0).(*tensor.Float32)
}
