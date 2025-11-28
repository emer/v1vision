// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

import (
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/gabor"
)

// AddGabor adds given [gabor.Filter] to Filters, returning the
// filter type index in Filters and the output values configured
// for storing the output of running these filters, per the
// given [Geom] output size. Adds a [ConvolveImage] operation
// for this Gabor filtering step, from given input image index.
func (vv *V1Vision) AddGabor(in int, gf *gabor.Filter, geom *Geom) (ftyp, out int) {
	ftyp = vv.NewFilter(gf.NAngles, gf.Size, gf.Size)
	vv.GaborToFilter(ftyp, gf)
	out = vv.NewConvolveImage(in, 0, ftyp, gf.NAngles, gf.Gain, geom)
	return
}

// GaborToFilter sets the given [gabor.Filter] filter to given
// filter type index. If more filters are added after AddGabor called
// then need to go back at the end and call all the ToFilter methods,
// in case the filters tensor has been resized.
func (vv *V1Vision) GaborToFilter(ftyp int, gf *gabor.Filter) {
	flt := vv.Filters.SubSpace(ftyp).(*tensor.Float32)
	gf.ToTensor(flt)
}
