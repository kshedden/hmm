package main

import (
	"compress/gzip"
	"encoding/gob"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/kshedden/multihmm/hmmlib"
	"github.com/kshedden/multihmm/hmmsim"
)

func main() {

	var obsmodel, varmodel, outname string
	flag.StringVar(&obsmodel, "obsmodel", "gaussian", "Observation distribution")
	flag.StringVar(&varmodel, "varmodel", "constant", "Variance model")
	flag.StringVar(&outname, "outname", "", "Output file name")

	var zeroinflated, masknull bool
	flag.BoolVar(&zeroinflated, "zeroinflated", false, "Zero inflated")
	flag.BoolVar(&masknull, "masknull", true, "Introduce null values at head and tail of sequence")

	var varpower, snr float64
	flag.Float64Var(&varpower, "varpower", 1.5, "Power variance for Tweedie")
	flag.Float64Var(&snr, "snr", 8, "Signal-to-noise ratio")

	var nParticleGrp, nState, nTime, nGroup int
	flag.IntVar(&nParticleGrp, "nparticlegrp", 0, "Number of particles per group")
	flag.IntVar(&nState, "nstate", 0, "Number of states")
	flag.IntVar(&nTime, "ntime", 0, "Number of time points")
	flag.IntVar(&nGroup, "ngroup", 0, "Number of groups")
	flag.Parse()

	rand.Seed(time.Now().UTC().UnixNano())

	hmm := hmmlib.NewMulti(nParticleGrp*nGroup, nState, nTime, nState, hmmlib.NoCollisionConstraintMaker)

	switch obsmodel {
	case "gaussian":
		hmm.ObsModel = hmmlib.Gaussian
	case "poisson":
		hmm.ObsModel = hmmlib.Poisson
	case "tweedie":
		hmm.ObsModel = hmmlib.Tweedie
		hmm.VarPower = varpower
	default:
		panic(fmt.Sprintf("generate: unknown obsmodel '%s'\n", obsmodel))
	}

	if hmm.ObsModel == hmmlib.Gaussian {
		switch varmodel {
		case "const":
			hmm.VarModel = hmmlib.VarConst
		case "constbystate":
			hmm.VarModel = hmmlib.VarConstByState
		case "constbycomponent":
			hmm.VarModel = hmmlib.VarConstByComponent
		case "free":
			hmm.VarModel = hmmlib.VarFree
		default:
			panic(fmt.Sprintf("generate: unknown varmodel '%s'\n", varmodel))
		}
	}

	hmm.ZeroInflated = zeroinflated

	if outname == "" {
		panic("'outname' is required")
	}

	// Put everyone into a group
	hmm.Group = make([][]int, nGroup)
	ii := 0
	for j := 0; j < nGroup; j++ {
		for k := 0; k < nParticleGrp; k++ {
			hmm.Group[j] = append(hmm.Group[j], ii)
			ii++
		}
	}

	// Set the transition matrix
	hmm.Trans = make([]float64, nState*nState)
	if hmm.NState == 1 {
		hmm.Trans = []float64{1}
	} else {
		for i := 0; i < nState; i++ {
			p := 0.8 + 0.1*float64(i)/float64(nState-1)
			for j := 0; j < nState; j++ {
				if i == j {
					hmm.Trans[i*nState+j] = p
				} else {
					hmm.Trans[i*nState+j] = (1 - p) / float64(nState-1)
				}
			}
		}
	}

	// Set the initial state probabilities
	hmm.Init = make([]float64, nState)
	for i := 0; i < hmm.NState; i++ {
		hmm.Init[i] = 1 / float64(nState)
	}

	// Set the parameters of the observation distribution
	hmm.Mean = make([]float64, nState*nState)
	for i := 0; i < nState; i++ {
		for j := 0; j < nState; j++ {
			switch hmm.ObsModel {
			case hmmlib.Gaussian:
				if i == j {
					hmm.Mean[i*nState+j] = snr
				}
			case hmmlib.Poisson:
				if i == j {
					hmm.Mean[i*nState+j] = snr
				} else {
					hmm.Mean[i*nState+j] = 1
				}
			case hmmlib.Tweedie:
				if i == j {
					hmm.Mean[i*nState+j] = snr
				} else {
					hmm.Mean[i*nState+j] = 1
				}
			default:
				panic("unkown obsmodel\n")
			}
		}
	}

	// Set the zero inflation probabilities if needed
	if hmm.ZeroInflated {
		hmm.Zprob = make([]float64, nState*nState)
		ii := 0
		for i := 0; i < hmm.NState; i++ {
			for j := 0; j < hmm.NState; j++ {
				if i == j {
					hmm.Zprob[ii] = 0.1
				} else {
					hmm.Zprob[ii] = 0.4
				}
				ii++
			}
		}
	}

	// Set the standard deviations if needed
	if hmm.ObsModel == hmmlib.Gaussian {
		hmm.Std = make([]float64, hmm.NState*hmm.NState)
		switch hmm.VarModel {
		case hmmlib.VarConst:
			for i := 0; i < hmm.NState; i++ {
				for j := 0; j < hmm.NState; j++ {
					hmm.Std[i*hmm.NState+j] = 1
				}
			}
		case hmmlib.VarConstByState:
			for i := 0; i < hmm.NState; i++ {
				for j := 0; j < hmm.NState; j++ {
					hmm.Std[i*hmm.NState+j] = 0.5 + float64(j)/float64(hmm.NState)
				}
			}
		case hmmlib.VarConstByComponent:
			for i := 0; i < hmm.NState; i++ {
				for j := 0; j < hmm.NState; j++ {
					hmm.Std[i*hmm.NState+j] = 0.5 + float64(i)/float64(hmm.NState)
				}
			}
		case hmmlib.VarFree:
			for i := 0; i < hmm.NState; i++ {
				for j := 0; j < hmm.NState; j++ {
					hmm.Std[i*hmm.NState+j] = 0.5 + float64(i*j)/float64((hmm.NState-1)*(hmm.NState-1))
				}
			}
		default:
			panic("Unkown variance model\n")
		}
	}

	hmmsim.GenStatesMulti(hmm, masknull)
	hmmsim.GenObs(&hmm.HMM)

	fid, err := os.Create(outname)
	if err != nil {
		panic(err)
	}
	defer fid.Close()

	gid := gzip.NewWriter(fid)
	defer gid.Close()

	enc := gob.NewEncoder(gid)

	if err := enc.Encode(&hmm); err != nil {
		panic(err)
	}
}
