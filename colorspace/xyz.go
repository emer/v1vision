// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colorspace

// SRGBLinToXYZ converts sRGB linear into XYZ CIE standard color space
func SRGBLinToXYZ(rl, gl, bl float32, x, y, z *float32) {
	*x = 0.4124*rl + 0.3576*gl + 0.1805*bl
	*y = 0.2126*rl + 0.7152*gl + 0.0722*bl
	*z = 0.0193*rl + 0.1192*gl + 0.9505*bl
}

// XYZToSRGBLin converts XYZ CIE standard color space to sRGB linear
func XYZToSRGBLin(x, y, z float32, rl, gl, bl *float32) {
	*rl = 3.2406*x + -1.5372*y + -0.4986*z
	*gl = -0.9689*x + 1.8758*y + 0.0415*z
	*bl = 0.0557*x + -0.2040*y + 1.0570*z
}

// SRGBToXYZ converts sRGB into XYZ CIE standard color space
func SRGBToXYZ(r, g, b float32, x, y, z *float32) {
	var rl, gl, bl float32
	SRGBToLinear(r, g, b, &rl, &gl, &bl)
	SRGBLinToXYZ(rl, gl, bl, x, y, z)
}

// XYZToSRGB converts XYZ CIE standard color space into sRGB
func XYZToSRGB(x, y, z float32, r, g, b *float32) {
	var rl, gl, bl float32
	XYZToSRGBLin(x, y, z, &rl, &gl, &bl)
	SRGBFromLinear(rl, bl, gl, r, g, b)
	return
}

// #CAT_ColorSpace renormalize XZY values relative to the D65 outdoor white light values
func XYZRenormD65(x, y, z float32, xr, yr, zr *float32) {
	*xr = x * (1 / 0.95047)
	*zr = z * (1 / 1.08883)
	*yr = y
	return
}
