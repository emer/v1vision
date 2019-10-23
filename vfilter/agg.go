// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"sync"

	"github.com/emer/etable/etensor"
	"github.com/emer/vision/nproc"
)

// FeatAgg does simple aggregation of feature rows from one feature map
// to another.  One row (inner-most of 4D dimensions) is assumed to be
// an angle, common across feature rows.
// srcRows is the list of rows in the source to copy.
// outStart is starting row in output to start copy -- srcRows will
// be contiguous in output from that row up.
// no bounds checking is done on output so it will just fail if
// there isn't enough room -- allocate the output size before calling!
func FeatAgg(srcRows []int, trgStart int, src, out *etensor.Float32) {
	nang := src.Dim(3)
	ncpu := nproc.NumCPU()
	nthrs, nper, rmdr := nproc.ThreadNs(ncpu, nang)
	var wg sync.WaitGroup
	for th := 0; th < nthrs; th++ {
		wg.Add(1)
		f := th * nper
		go featAggThr(&wg, f, nper, srcRows, trgStart, src, out)
	}
	if rmdr > 0 {
		wg.Add(1)
		f := nthrs * nper
		go featAggThr(&wg, f, rmdr, srcRows, trgStart, src, out)
	}
	wg.Wait()
}

// featAggThr is per-thread implementation
func featAggThr(wg *sync.WaitGroup, fno, nf int, srcRows []int, trgStart int, src, out *etensor.Float32) {
	ny := src.Dim(0)
	nx := src.Dim(1)
	for fi := 0; fi < nf; fi++ {
		ang := fno + fi
		for y := 0; y < ny; y++ {
			for x := 0; x < nx; x++ {
				for si, sr := range srcRows {
					sv := src.Value([]int{y, x, sr, ang})
					out.Set([]int{y, x, trgStart + si, ang}, sv)
				}
			}
		}
	}
	wg.Done()
}