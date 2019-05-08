// This is a series of tests to conmfirm that the log-likelihood is non-decreasing over the EM iterations.

package hmmlib

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

const (
	niter = 20
)

func gendat(npg, ngp, nst, ntm, ncp int) *HMM {

	hmm := New(npg*ngp, nst, ntm, ncp)

	hmm.Obs = make([][]float64, npg*ngp)
	for k := 0; k < ngp*npg; k++ {
		obs := make([]float64, ncp*ntm)
		for j := range obs {
			obs[j] = math.Floor(15 * rand.Float64())
		}

		// Introduce some null states at the end
		for j := 0; j < rand.Int()%4; j++ {
			ii := (ntm - 1 - j) * hmm.NComp
			for q := 0; q < hmm.NComp; q++ {
				obs[ii+q] = NullObs
			}
		}

		// Introduce some null states at the beginning
		for j := 0; j < rand.Int()%4; j++ {
			ii := (k%3 + j) * hmm.NComp
			for q := 0; q < hmm.NComp; q++ {
				obs[ii+q] = NullObs
			}
		}

		hmm.Obs[k] = obs
	}

	hmm.SetLogger("t")
	hmm.Initialize()
	hmm.SetStartParams()

	return hmm
}

func TestLLFTweedie(t *testing.T) {

	for _, ngp := range []int{2, 5, 10} {
		for _, npg := range []int{5, 10, 20} {
			for _, nst := range []int{2, 4, 8} {
				for _, ntm := range []int{10, 20, 30} {
					for _, vp := range []float64{1.1, 1.3, 1.5, 1.7, 1.9} {
						for _, ncp := range []int{1, 2, 4} {

							hmm := gendat(ngp, npg, nst, ntm, ncp)
							hmm.ObsModel = Tweedie
							hmm.VarPower = vp
							hmm.Fit(niter)

							// Check that the log-likelihood values are ascending.
							for i := 1; i < len(hmm.LLF); i++ {
								if hmm.LLF[i] < hmm.LLF[i-1] {
									fmt.Printf("iter=%d\n", i)
									fmt.Printf("%f %f %f\n", hmm.LLF[i-1], hmm.LLF[i], hmm.LLF[i-1]-hmm.LLF[i])
									t.Fail()
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestLLFPoisson(t *testing.T) {

	for _, ngp := range []int{2, 5, 10} {
		for _, npg := range []int{5, 10, 20} {
			for _, nst := range []int{2, 4, 8} {
				for _, ntm := range []int{10, 20, 30} {
					for _, ncp := range []int{1, 2, 4} {

						hmm := gendat(ngp, npg, nst, ntm, ncp)
						hmm.ObsModel = Poisson
						hmm.Fit(niter)

						// Check that the log-likelihood values are ascending.
						for i := 1; i < len(hmm.LLF); i++ {
							if hmm.LLF[i] < hmm.LLF[i-1] {
								fmt.Printf("iter=%d\n", i)
								fmt.Printf("%f %f %f\n", hmm.LLF[i-1], hmm.LLF[i], hmm.LLF[i-1]-hmm.LLF[i])
								t.Fail()
							}
						}
					}
				}
			}
		}
	}
}

func TestLLFGaussian(t *testing.T) {

	for _, ngp := range []int{2, 5, 10} {
		for _, npg := range []int{5, 10, 20} {
			for _, nst := range []int{2, 4, 8} {
				for _, ntm := range []int{10, 20, 30} {
					for _, vm := range []VarModelType{VarFree, VarConst, VarConstByState, VarConstByComponent} {

						hmm := gendat(ngp, npg, nst, ntm, nst)
						hmm.ObsModel = Gaussian
						hmm.VarModel = vm
						hmm.Fit(niter)

						// Check that the log-likelihood values are ascending.
						for i := 1; i < len(hmm.LLF); i++ {
							if hmm.LLF[i] < hmm.LLF[i-1] {
								fmt.Printf("VarModel %d, iter=%d\n", vm, i)
								fmt.Printf("%f %f %f\n", hmm.LLF[i-1], hmm.LLF[i], hmm.LLF[i-1]-hmm.LLF[i])
								t.Fail()
								panic("")
							}
						}
					}
				}
			}
		}
	}
}
