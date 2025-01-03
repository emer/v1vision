// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"image"
	"math/rand"
	"sync"

	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/nproc"
)

// UnPool performs inverse max-pooling over given pool size and spacing.
// This is very dumb and either uses a random number if rnd = true, or
// just copies the max pooled value over all of the
// individual elements that were pooled.  A smarter solution would require
// maintaining the index of the max item, but that requires more infrastructure
// size must = spacing or 2 * spacing.
// Pooling is sensitive to the feature structure of the input, which
// must have shape: Y, X, Polarities, Angles.
func UnPool(psize, spc image.Point, in, out *tensor.Float32, rnd bool) {
	ny := in.DimSize(0)
	nx := in.DimSize(1)
	pol := in.DimSize(2)
	nang := in.DimSize(3)
	oy := ny / int(spc.Y)
	ox := nx / int(spc.X)
	if spc.Y != psize.Y {
		oy--
	}
	if spc.X != psize.X {
		ox--
	}

	out.SetShapeSizes(oy, ox, pol, nang)
	nf := pol * nang
	ncpu := nproc.NumCPU()
	nthrs, nper, rmdr := nproc.ThreadNs(ncpu, nf)
	var wg sync.WaitGroup
	for th := 0; th < nthrs; th++ {
		wg.Add(1)
		f := th * nper
		go unPoolThr(&wg, f, nper, psize, spc, in, out, rnd)
	}
	if rmdr > 0 {
		wg.Add(1)
		f := nthrs * nper
		go unPoolThr(&wg, f, rmdr, psize, spc, in, out, rnd)
	}
	wg.Wait()
}

// unPoolThr is per-thread implementation
func unPoolThr(wg *sync.WaitGroup, fno, nf int, psize, spc image.Point, in, out *tensor.Float32, rnd bool) {
	ny := out.DimSize(0)
	nx := out.DimSize(1)
	nang := out.DimSize(3)
	psz := psize.X * psize.Y
	for fi := 0; fi < nf; fi++ {
		f := fno + fi
		pol := f / nang
		ang := f % nang
		for y := 0; y < ny; y++ {
			iy := y * spc.Y
			for x := 0; x < nx; x++ {
				ix := x * spc.X
				mx := out.Value(y, x, pol, ang)
				if rnd {
					ptrg := rand.Intn(psz)
					pdx := 0
					for py := 0; py < psize.Y; py++ {
						for px := 0; px < psize.X; px++ {
							if pdx == ptrg {
								in.Set(mx, iy+py, ix+px, pol, ang)
							} else {
								in.Set(0, iy+py, ix+px, pol, ang)
							}
							pdx++
						}
					}
				} else {
					for py := 0; py < psize.Y; py++ {
						for px := 0; px < psize.X; px++ {
							in.Set(mx, iy+py, ix+px, pol, ang)
						}
					}
				}
			}
		}
	}
	wg.Done()
}
