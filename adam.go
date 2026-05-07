// adam.go — Adam optimizer with beta1=0.85, beta2=0.99 (as in microgpt)
package main

import "math"

type Adam struct {
	m, v  []float64
	t     int
	beta1 float64
	beta2 float64
	eps   float64
}

func NewAdam(nParams int) *Adam {
	return &Adam{
		m:     make([]float64, nParams),
		v:     make([]float64, nParams),
		beta1: 0.9,
		beta2: 0.999,
		eps:   1e-8,
	}
}

func (a *Adam) Update(params, grads []float64, lr float64) {
	a.t++
	for i := range params {
		a.m[i] = a.beta1*a.m[i] + (1-a.beta1)*grads[i]
		a.v[i] = a.beta2*a.v[i] + (1-a.beta2)*grads[i]*grads[i]
		mHat := a.m[i] / (1 - math.Pow(a.beta1, float64(a.t)))
		vHat := a.v[i] / (1 - math.Pow(a.beta2, float64(a.t)))
		params[i] -= lr * mHat / (math.Sqrt(vHat) + a.eps)
	}
}
