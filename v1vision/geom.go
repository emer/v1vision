// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

import (
	"image"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/gosl/slvec"
)

//gosl:start

// Geom contains the filtering geometry info for a given filter pass.
type Geom struct {

	// size of input -- computed from image or set
	In slvec.Vector2i

	// size of output -- computed
	Out slvec.Vector2i

	// starting border into image -- must be >= FilterRt
	Border slvec.Vector2i

	// spacing -- number of pixels to skip in each direction
	Spacing slvec.Vector2i

	// full size of filter
	FilterSz slvec.Vector2i

	// computed size of left/top size of filter
	FilterLt slvec.Vector2i

	// computed size of right/bottom size of filter (FilterSz - FilterLeft)
	FilterRt slvec.Vector2i
}

//gosl:end

// Set sets the basic geometry params
func (ge *Geom) Set(border, spacing, filtSz math32.Vector2i) {
	ge.Border.SetV(border)
	ge.Spacing.SetV(spacing)
	ge.FilterSz.SetV(filtSz)
	ge.UpdtFilter()
}

// LeftHalf returns the left / top half of a filter
func LeftHalf(x int32) int32 {
	if x%2 == 0 {
		return x / 2
	}
	return (x - 1) / 2
}

// UpdtFilter updates filter sizes, and ensures that Border >= FilterRt
func (ge *Geom) UpdtFilter() {
	ge.FilterLt.X = LeftHalf(ge.FilterSz.X)
	ge.FilterLt.Y = LeftHalf(ge.FilterSz.Y)
	ge.FilterRt.SetV(ge.FilterSz.V().Sub(ge.FilterLt.V()))
	if ge.Border.X < ge.FilterRt.X {
		ge.Border.X = ge.FilterRt.X
	}
	if ge.Border.Y < ge.FilterRt.Y {
		ge.Border.Y = ge.FilterRt.Y
	}
}

// SetImageSize sets the original image input size that excludes
// the border size, so this adds 2* border to that for the total
// input size.
func (ge *Geom) SetImageSize(imSize image.Point) {
	in := math32.Vec2i(imSize.X, imSize.Y)
	b2 := ge.Border.V().MulScalar(2)
	av := in.Add(b2)
	ge.In.SetV(av)
	ge.Out.SetV(in.DivScalar(ge.Spacing.X))
}

// SetInputSize sets the input size, and computes output from that.
func (ge *Geom) SetInputSize(inSize image.Point) {
	ge.In.X = int32(inSize.X)
	ge.In.Y = int32(inSize.Y)
	b2 := ge.Border.V().MulScalar(2)
	av := ge.In.V().Sub(b2)
	ge.Out.SetV(av.DivScalar(ge.Spacing.X)) // only 1
}
