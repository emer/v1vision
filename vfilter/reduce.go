// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"sync"

	"cogentcore.org/core/tensor"
	"github.com/emer/v1vision/nproc"
)

// MaxReduceFilterY performs max-pooling reduce over inner Filter Y
// dimension (polarities, colors)
// must have shape: Y, X, Polarities, Angles.
func MaxReduceFilterY(in, out *tensor.Float32) {
	ny := in.DimSize(0)
	nx := in.DimSize(1)
	nang := in.DimSize(3)
	out.SetShapeSizes(ny, nx, 1, nang)
	ncpu := nproc.NumCPU()
	nthrs, nper, rmdr := nproc.ThreadNs(ncpu, nang)
	var wg sync.WaitGroup
	for th := 0; th < nthrs; th++ {
		wg.Add(1)
		f := th * nper
		go maxReduceFilterYThr(&wg, f, nper, in, out)
	}
	if rmdr > 0 {
		wg.Add(1)
		f := nthrs * nper
		go maxReduceFilterYThr(&wg, f, rmdr, in, out)
	}
	wg.Wait()
}

// maxReduceFilterYThr is per-thread implementation
func maxReduceFilterYThr(wg *sync.WaitGroup, fno, nf int, in, out *tensor.Float32) {
	ny := in.DimSize(0)
	nx := in.DimSize(1)
	np := in.DimSize(2)
	for fi := 0; fi < nf; fi++ {
		ang := fno + fi
		for y := 0; y < ny; y++ {
			for x := 0; x < nx; x++ {
				mx := float32(0)
				for fy := 0; fy < np; fy++ {
					iv := in.Value(y, x, fy, ang)
					if iv > mx {
						mx = iv
					}
				}
				out.Set(mx, y, x, 0, ang)
			}
		}
	}
	wg.Done()
}
