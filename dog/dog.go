// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
package dog provides the Difference-of-Gaussians (DoG) filter for visual and other
forms of signal processing
*/
package dog

//go:generate core generate -add-types

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
)

// dog.Filter specifies a DoG Difference of Gaussians filter function.
type Filter struct {

	// is this filter active?
	On bool

	// how much relative weight does this filter have when combined with other filters
	Wt float32

	// overall gain multiplier applied after dog filtering -- only relevant if not using renormalization (otherwize it just gets renormed away)
	Gain float32 `default:"8"`

	// gain for the on component of filter, only relevant for color-opponent DoG's
	OnGain float32 `default:"1"`

	// size of the overall filter -- number of pixels wide and tall for a square matrix used to encode the filter -- filter is centered within this square -- typically an even number, min effective size ~6
	Size int

	// how far apart to space the centers of the dog filters -- 1 = every pixel, 2 = every other pixel, etc -- high-res should be 1 or 2, lower res can be increments therefrom
	Spacing int

	// gaussian sigma for the narrower On gaussian, in normalized units relative to Size
	OnSig float32 `default:"0.125"`

	// gaussian sigma for the wider Off gaussian, in normalized units relative to Size
	OffSig float32 `default:"0.25"`

	// cut off the filter (to zero) outside a circle of diameter = Size -- makes the filter more radially symmetric
	CircleEdge bool `default:"true"`
}

func (gf *Filter) Defaults() {
	gf.On = true
	gf.Wt = 1
	gf.Gain = 8
	gf.OnGain = 1
	gf.Size = 12
	gf.Spacing = 2
	gf.OnSig = 0.125
	gf.OffSig = 0.25
	gf.CircleEdge = true
}

func (gf *Filter) Update() {
}

func (gf *Filter) ShouldDisplay(field string) bool {
	switch field {
	case "On":
		return true
	default:
		return gf.On
	}
}

// SetSize sets the size and spacing -- these are the main params
// that need to be varied for standard V1 dogs.
func (gf *Filter) SetSize(sz, spc int) {
	gf.Size = sz
	gf.Spacing = spc
}

// GaussDenSig returns gaussian density for given value and sigma
func GaussDenSig(x, sig float32) float32 {
	x /= sig
	return 0.398942280 * math32.Exp(-0.5*x*x) / sig
}

// ToTensor renders dog filters into the given table tensor.Tensor,
// setting dimensions to [3][Y][X] where Y = X = Size, and
// first one is On-filter, second is Off-filter, and third is Net On - Off
func (gf *Filter) ToTensor(tsr *tensor.Float32) {
	tsr.SetShapeSizes(int(FiltersN), gf.Size, gf.Size)

	ctr := 0.5 * float32(gf.Size-1)
	radius := float32(gf.Size) * 0.5

	gsOn := gf.OnSig * float32(gf.Size)
	gsOff := gf.OffSig * float32(gf.Size)

	var posSum, negSum, onSum, offSum float32
	for y := 0; y < gf.Size; y++ {
		for x := 0; x < gf.Size; x++ {
			xf := float32(x) - ctr
			yf := float32(y) - ctr

			dist := math32.Hypot(xf, yf)
			var ong, offg float32
			if !(gf.CircleEdge && (dist > radius)) {
				ong = GaussDenSig(dist, gsOn)
				offg = GaussDenSig(dist, gsOff)
			}
			tsr.Set(ong, int(On), y, x)
			tsr.Set(offg, int(Off), y, x)
			onSum += ong
			offSum += offg
			net := ong - offg
			tsr.Set(net, int(Net), y, x)
			if net > 0 {
				posSum += net
			} else if net < 0 {
				negSum += -net
			}
		}
	}
	// renorm each half, separate components
	for y := 0; y < gf.Size; y++ {
		for x := 0; x < gf.Size; x++ {
			val := tsr.Value(int(Net), y, x)
			if val > 0 {
				val /= posSum
			} else if val < 0 {
				val /= negSum
			}
			tsr.Set(val, int(Net), y, x)
			on := tsr.Value(int(On), y, x)
			tsr.Set(on/onSum, int(On), y, x)
			off := tsr.Value(int(Off), y, x)
			tsr.Set(off/offSum, int(Off), y, x)
		}
	}
}

// ToTable renders filters into the given table.Table
// setting a column named Version and  a column named Filter
// to the filter for that version (on, off, net)
// This is useful for display and validation purposes.
func (gf *Filter) ToTable(tab *table.Table) {
	tab.AddStringColumn("Version")
	tab.AddFloat32Column("Filter", int(FiltersN), gf.Size, gf.Size)
	tab.SetNumRows(3)
	gf.ToTensor(tab.Columns.Values[1].(*tensor.Float32))
	nm := tab.ColumnByIndex(0)
	nm.SetString("On", int(On))
	nm.SetString("Off", int(Off))
	nm.SetString("Net", int(Net))
}

// FilterTensor extracts the given filter subspace from set of 3 filters in input tensor
// 0 = On, 1 = Off, 2 = Net
func (gf *Filter) FilterTensor(tsr *tensor.Float32, filt Filters) *tensor.Float32 {
	return tsr.SubSpace(int(filt)).(*tensor.Float32)
}

// Filters is the type of filter
type Filters int

const (
	On Filters = iota
	Off
	Net
	FiltersN
)
