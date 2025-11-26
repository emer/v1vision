// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

import (
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/dog"
)

// AddDoG adds
func (vv *V1Vision) AddDoG(in int, df *dog.Filter, geom *Geom) (ftyp, out int) {
	ftyp = vv.NewFilter(1, df.Size, df.Size)
	flt := vv.Filters.SubSpace(ftyp).(*tensor.Float32)
	df.ToTensor(flt, true) // only net
	out = vv.NewValues(int(geom.Out.Y), int(geom.Out.X), 1)
	vv.NewConvolveImage(in, 0, out, ftyp, 1, df.Gain, geom)
	return
}
