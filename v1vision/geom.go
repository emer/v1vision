// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

import (
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

	// starting border into image -- must be >= FiltRt
	Border slvec.Vector2i

	// spacing -- number of pixels to skip in each direction
	Spacing slvec.Vector2i

	// full size of filter
	FiltSz slvec.Vector2i

	// computed size of left/top size of filter
	FiltLt slvec.Vector2i

	// computed size of right/bottom size of filter (FiltSz - FiltLeft)
	FiltRt slvec.Vector2i
}

//gosl:end

// Set sets the basic geometry params
func (ge *Geom) Set(border, spacing, filtSz math32.Vector2i) {
	ge.Border.SetV(border)
	ge.Spacing.SetV(spacing)
	ge.FiltSz.SetV(filtSz)
	ge.UpdtFilt()
}

// LeftHalf returns the left / top half of a filter
func LeftHalf(x int32) int32 {
	if x%2 == 0 {
		return x / 2
	}
	return (x - 1) / 2
}

// UpdtFilt updates filter sizes, and ensures that Border >= FiltRt
func (ge *Geom) UpdtFilt() {
	ge.FiltLt.X = LeftHalf(ge.FiltSz.X)
	ge.FiltLt.Y = LeftHalf(ge.FiltSz.Y)
	ge.FiltRt.SetV(ge.FiltSz.V().Sub(ge.FiltLt.V()))
	if ge.Border.X < ge.FiltRt.X {
		ge.Border.X = ge.FiltRt.X
	}
	if ge.Border.Y < ge.FiltRt.Y {
		ge.Border.Y = ge.FiltRt.Y
	}
}

// SetSize sets the input size, and computes output from that.
func (ge *Geom) SetSize(inSize math32.Vector2i) {
	ge.In.SetV(inSize)
	b2 := ge.Border.V().MulScalar(2)
	av := ge.In.V().Sub(b2)
	ge.Out.SetV(av.DivScalar(ge.Spacing.X)) // only 1
}
