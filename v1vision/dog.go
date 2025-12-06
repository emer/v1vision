// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1vision

import (
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/dog"
)

// NewDoG adds given [dog.Filter] Net filter to Filters, for
// spatial contrast filtering.
// Returns the filter type index in Filters and the output values
// configured for storing the output of running these filters,
// per the given [Geom] output size. Adds a [ConvolveImage] operation
// for this DoG filtering step, from given input image index,
// and irgb color channel (0-2).
func (vv *V1Vision) NewDoG(in, irgb int, df *dog.Filter, geom *Geom) (ftyp, out int) {
	ftyp = vv.NewFilter(1, df.Size, df.Size)
	vv.DoGToFilter(ftyp, df)
	out = vv.NewConvolveImage(in, irgb, ftyp, 1, df.Gain, geom)
	return
}

// DoGToFilter sets the given [dog.Filter] Net filter to given
// filter type index. If more filters are added after NewDoG is called
// then need to go back at the end and call all the ToFilter methods,
// in case the filters tensor has been resized.
func (vv *V1Vision) DoGToFilter(ftyp int, df *dog.Filter) {
	flt := vv.Filters.SubSpace(ftyp).(*tensor.Float32)
	df.ToTensor(flt, dog.Net)
}

// NewDoGOnOff adds given [dog.Filter] On and Off filters to Filters,
// for color contrast filtering, applying On and Off to different color
// channels. Returns the filter type index.
func (vv *V1Vision) NewDoGOnOff(df *dog.Filter, geom *Geom) int {
	ftyp := vv.NewFilter(2, df.Size, df.Size)
	vv.DoGOnOffToFilter(ftyp, df)
	return ftyp
}

// DoGOnOffToFilter sets the given [dog.Filter] On and Off filters to given
// filter type index. If more filters are added after NewDoGOnOff is called
// then need to go back at the end and call all the ToFilter methods,
// in case the filters tensor has been resized.
func (vv *V1Vision) DoGOnOffToFilter(ftyp int, df *dog.Filter) {
	flt := vv.Filters.SubSpace(ftyp).(*tensor.Float32)
	df.ToTensor(flt, dog.On, dog.Off)
}
