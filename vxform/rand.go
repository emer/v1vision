// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vxform

import (
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/lab/base/randx"
)

// Rand specifies random transforms
type Rand struct {

	// min -- max range of X-axis (horizontal) translations to generate (as proportion of image size)
	TransX minmax.F32

	// min -- max range of Y-axis (vertical) translations to generate (as proportion of image size)
	TransY minmax.F32

	// min -- max range of scales to generate
	Scale minmax.F32

	// min -- max range of rotations to generate (in degrees)
	Rot minmax.F32
}

// Gen Generates new random transform values
func (rx *Rand) Gen(xf *XForm, rnd *randx.SysRand) {
	trX := rx.TransX.ProjValue(rnd.Float32())
	trY := rx.TransY.ProjValue(rnd.Float32())
	sc := rx.Scale.ProjValue(rnd.Float32())
	rt := rx.Rot.ProjValue(rnd.Float32())
	xf.Set(trX, trY, sc, rt)
}
