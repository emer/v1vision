// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

import (
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/dog"
)

// AddDoG adds given [dog.Filter] to Filters, returning the
// filter type index in Filters and the output values configured
// for storing the output of running these filters, per the
// given [Geom] output size. Adds a [ConvolveImage] operation
// for this DoG filtering step, from given input image index.
func (vv *V1Vision) AddDoG(in int, df *dog.Filter, geom *Geom) (ftyp, out int) {
	ftyp = vv.NewFilter(1, df.Size, df.Size)
	vv.DoGToFilter(ftyp, df)
	out = vv.NewValues(int(geom.Out.Y), int(geom.Out.X), 1)
	vv.NewConvolveImage(in, 0, out, ftyp, 1, df.Gain, geom)
	return
}

// DoGToFilter sets the given [dog.Filter] filter to given
// filter type index. If more filters are added after AddDoG called
// then need to go back at the end and call all the ToFilter methods,
// in case the filters tensor has been resized.
func (vv *V1Vision) DoGToFilter(ftyp int, df *dog.Filter) {
	flt := vv.Filters.SubSpace(ftyp).(*tensor.Float32)
	df.ToTensor(flt, true) // only net
}
