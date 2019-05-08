//

package hmmlib

import (
	"bytes"
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"sync"

	"gonum.org/v1/gonum/floats"
)

const (
	// Maximum allowed value for Zprob
	zpmax float64 = 0.95

	// The Poisson/Tweedie mean parameters are never allowed to go below this value
	minPoissonMean = 1e-8

	// Minimum allowed value for the observation SD
	sdmin = 1e-8

	// Null value for an observation.  When the observed value is Null, it is
	// not used in estimation.  If the first component of the observed vector
	// is Null, the are all treated as Null.
	NullObs float64 = -9999

	// Null value for a state
	NullState int = -9999
)

// VarModelType indicates how the variance structure is modeled.  This is only
// used for Gaussian emission models.
type VarModelType uint8

// VarFree, VarConst are the available variance models.
const (
	VarFree VarModelType = iota
	VarConst
	VarConstByState
	VarConstByComponent
)

// ObsModelType indicates the emission model distribution.
type ObsModelType uint8

// Gaussian, etc. are the available emission models.
const (
	Gaussian ObsModelType = iota
	Poisson
	Tweedie
)

// HMM represents a hidden Markov model for a collection of particles
// that follow the same HMM law.
type HMM struct {

	// Number of particles (e.g. subjects)
	NParticle int

	// Number of time points
	NTime int

	// Number of states
	NState int

	// Number of components of the emission vector
	NComp int

	// The type of variance constraint
	VarModel VarModelType

	// The observation distribution
	ObsModel ObsModelType

	// If true the model is zero-inflated
	ZeroInflated bool

	// The transition probability matrix
	Trans []float64

	// The initial probability distribution
	Init []float64

	// The observation means. If ZeroInflated=true, these are the observation means
	// given that they are not zero.
	Mean []float64

	// The observation standard deviations.  If ZeroInflated=true, these are the
	// observation means given that they are not zero.
	Std []float64

	// The log probability of observing zero, only used if ZeroInflated=true.
	Zprob []float64

	// The true states (if known)
	State [][]int

	// The observations
	Obs [][]float64

	// The forward probabilities
	Fprob [][]float64

	// The backward probabilities
	Bprob [][]float64

	// The reconstructed states
	PState [][]int

	// The overall log-likelihood
	LLF []float64

	// The log-likelihood function for one particle
	llf []float64

	// The Tweedie variance is mean^VarPower
	VarPower float64

	Warnings warnings

	// Write log messages here
	msglogger *log.Logger
	parlogger *log.Logger
}

type warnings struct {
	TweedieUpperTail      int
	LogLikeDecreased      int
	SDTruncate            int
	NormalizeMaxThreshold int
}

// New returns an HMM value with the given size parameters.
func New(NParticle, NState, NTime, NComp int) *HMM {

	hmm := &HMM{
		NParticle: NParticle,
		NTime:     NTime,
		NState:    NState,
		NComp:     NComp,
	}

	return hmm
}

// SetLog provides a logger that will be used to write logging messages.
func (hmm *HMM) SetLogger(logname string) *log.Logger {

	fid, err := os.Create(logname + "_msg.log")
	if err != nil {
		panic(err)
	}
	hmm.msglogger = log.New(fid, "", log.Ltime)

	fid, err = os.Create(logname + "_par.log")
	if err != nil {
		panic(err)
	}
	hmm.parlogger = log.New(fid, "", 0)

	// The calling program can also use this logger
	return hmm.msglogger
}

// Initialize allocates workspaces for parameter estimation.  Call
// this prior to calling Fit.
func (hmm *HMM) Initialize() {

	if hmm.ZeroInflated && hmm.ObsModel != Gaussian {
		fmt.Printf("When ZeroInflated is true, ObsModelForm must be Gaussian")
		panic("")
	}

	if hmm.ObsModel != Gaussian {
		for _, obs := range hmm.Obs {
			for _, v := range obs {
				if v != NullObs && v < 0 {
					panic("Can't have negative observed values with non-Gaussian emission model")
				}
			}
		}
	}

	hmm.Bprob = makeFloatArray(hmm.NParticle, hmm.NState*hmm.NTime)
	hmm.Fprob = makeFloatArray(hmm.NParticle, hmm.NState*hmm.NTime)
	hmm.llf = make([]float64, hmm.NParticle)

	if hmm.msglogger == nil {
		hmm.msglogger = log.New(os.Stderr, "", log.Ltime)
	}

	hmm.msglogger.Printf("%d particles\n", hmm.NParticle)
	hmm.msglogger.Printf("%d components per emission\n", hmm.NComp)
	hmm.msglogger.Printf("%d time points per particle\n", hmm.NTime)
	hmm.msglogger.Printf("%d states\n", hmm.NState)
}

// ReconstructStates uses the Viterbi algorithm to predict
// the sequence of states for each particle.  The algorithm
// is run separately for each particle.  The reconstructed
// states are written into the PState component of the HMM.
func (hmm *HMM) ReconstructStates() {

	hmm.PState = makeIntArray(hmm.NParticle, hmm.NTime)

	for p := 0; p < hmm.NParticle; p++ {
		hmm.ReconstructParticle(p)
	}
}

// ReconstructParticle uses the Viterbi algorithm to predict
// the sequence of states for one particle.
func (hmm *HMM) ReconstructParticle(p int) {

	lpr := make([]float64, hmm.NTime*hmm.NState)
	lpt := make([]int, hmm.NTime*hmm.NState)

	hmm.reconstructionProbs(p, lpr, lpt)
	hmm.traceback(p, lpr, lpt)
}

func (hmm *HMM) reconstructionProbs(p int, lpr []float64, lpt []int) {

	obs := hmm.Obs[p]
	wk := make([]float64, hmm.NState)

	// Construct the table of conditional probabilities
	j0 := -2 * hmm.NState
	j1 := -hmm.NState
	k0 := -2 * hmm.NComp
	k1 := -hmm.NComp
	for t := 0; t < hmm.NTime; t++ {

		j0 += hmm.NState
		j1 += hmm.NState
		k0 += hmm.NComp
		k1 += hmm.NComp

		if obs[k1] == NullObs {
			continue
		}

		// Beginning from initial conditions
		if t == 0 || obs[k0] == NullObs {
			for st := 0; st < hmm.NState; st++ {
				lpr[j1+st] = hmm.GetLogObsProb(p, t, st, false) + math.Log(hmm.Init[st])
			}
			continue // First block of lpt is not used
		}

		// From st1 to st2
		for st2 := 0; st2 < hmm.NState; st2++ {
			for st1 := 0; st1 < hmm.NState; st1++ {
				wk[st1] = lpr[j0+st1] + math.Log(hmm.Trans[st1*hmm.NState+st2])
			}

			// The best previous state
			jj := argmax(wk)
			lpt[j1+st2] = jj
			lpr[j1+st2] = wk[jj] + hmm.GetLogObsProb(p, t, st2, false)
		}
	}
}

// Message writes a message to the message log.
func (hmm *HMM) Message(msg string) {
	hmm.msglogger.Print(msg)
}

func (hmm *HMM) traceback(p int, lpr []float64, lpt []int) {

	obs := hmm.Obs[p]
	y := hmm.PState[p]
	jt := hmm.NState * hmm.NTime
	kt := hmm.NComp * hmm.NTime

	for t := hmm.NTime - 1; t >= 0; t-- {

		jt -= hmm.NState
		kt -= hmm.NComp

		if obs[kt] == NullObs {
			y[t] = NullState
			continue
		}

		// Starting a new block
		if t == len(y)-1 || (obs[kt] != NullObs && obs[kt+hmm.NComp] == NullObs) {
			a := t * hmm.NState
			y[t] = argmax(lpr[a : a+hmm.NState])
			continue
		}

		y[t] = lpt[jt+hmm.NState+y[t+1]]
	}
}

// SetStartParams sets the starting parameters for the EM (Baum-Welch)
// optimization.
func (hmm *HMM) SetStartParams() {

	mean, std := hmm.MarginalMoments()

	hmm.Mean = make([]float64, hmm.NState*hmm.NComp)
	mn := floats.Sum(mean) / float64(len(mean))
	for i := 0; i < hmm.NState; i++ {
		for j := 0; j < hmm.NComp; j++ {
			if i == j {
				hmm.Mean[i*hmm.NComp+j] = mn
			} else {
				hmm.Mean[i*hmm.NComp+j] = mn / 10
			}
		}
	}

	if hmm.ObsModel == Gaussian {
		hmm.Std = make([]float64, hmm.NState*hmm.NComp)
		sdmn := floats.Sum(std) / float64(len(std))
		for i := 0; i < hmm.NState; i++ {
			for j := 0; j < hmm.NComp; j++ {
				hmm.Std[i*hmm.NComp+j] = sdmn
			}
		}
	}

	if hmm.ZeroInflated {
		hmm.Zprob = make([]float64, hmm.NState*hmm.NComp)
		for i := 0; i < hmm.NState; i++ {
			for j := 0; j < hmm.NComp; j++ {
				hmm.Zprob[i*hmm.NComp+j] = 0.1
			}
		}
	}

	hmm.Trans = make([]float64, hmm.NState*hmm.NState)
	for i := 0; i < hmm.NState; i++ {
		for j := 0; j < hmm.NState; j++ {
			if i == j {
				hmm.Trans[i*hmm.NState+j] = 0.8
			} else {
				hmm.Trans[i*hmm.NState+j] = 0.2 / float64(hmm.NState-1)
			}
		}
	}

	hmm.Init = make([]float64, hmm.NState)
	for i := 0; i < hmm.NState; i++ {
		hmm.Init[i] = 1 / float64(hmm.NState)
	}
}

// GetLogObsProb returns the probability P(obs | state=st, time=t), where
// obs is an observed (emitted) value from the HMM.  Factors depending
// only on 'obs' are omitted.
func (hmm *HMM) GetLogObsProb(p, t, st int, exact bool) float64 {

	switch hmm.ObsModel {
	case Gaussian:
		return hmm.getGaussianLogObsProb(p, t, st, exact)
	case Poisson:
		return hmm.getPoissonLogObsProb(p, t, st, exact)
	case Tweedie:
		return hmm.getTweedieLogObsProb(p, t, st, exact)
	default:
		panic("unknown model")
	}
}

// lgamma returns the log of the gamma function for a positive argument.
func lgamma(x float64) float64 {
	u, _ := math.Lgamma(x)
	return u
}

func (hmm *HMM) getTweedieLogObsProb(p, t, st int, exact bool) float64 {

	// http://www.statsci.org/smyth/pubs/tweediepdf-series-preprint.pdf

	obs := hmm.Obs[p]

	if obs[t*hmm.NComp] == NullObs {
		panic("should not reach here")
	}

	pw := hmm.VarPower

	var lpr float64
	ii := st * hmm.NComp
	for j := 0; j < hmm.NComp; j++ {
		y := obs[t*hmm.NComp+j]
		mn := hmm.Mean[ii]
		if mn < minPoissonMean {
			mn = minPoissonMean
		}
		lmn := math.Log(mn)
		lpr += y*math.Exp((1-pw)*lmn)/(1-pw) - math.Exp((2-pw)*lmn)/(2-pw)
		ii++
	}

	if !exact {
		return lpr
	}

	// calculate the Tweedie scaling factor here
	alp := float64(2-pw) / float64(1-pw)
	scale := 1.0
	lscale := math.Log(scale)
	for j := 0; j < hmm.NComp; j++ {
		y := obs[t*hmm.NComp+j]

		// Scaling factor is 1 in in this case
		if y == 0 {
			continue
		}

		lz := -alp*math.Log(y) + alp*math.Log(pw-1) - math.Log(2-pw) - (1-alp)*lscale
		kf := math.Pow(y, 2-pw) / (scale * float64(2-pw))
		k := int(math.Round(kf))
		if k < 1 {
			k = 1
		}

		// Sum the upper tail.
		w0 := float64(k)*lz - lgamma(float64(k+1)) - lgamma(-alp*float64(k))
		ws := 1.0
		for j := k + 1; j < 200; j++ {
			w1 := float64(j)*lz - lgamma(float64(j+1)) - lgamma(-alp*float64(j))
			if w1 < w0-37 {
				break
			}
			ws += math.Exp(w1 - w0)
			if j > 198 {
				hmm.Warnings.TweedieUpperTail++
			}
		}

		// Sum the lower tail.
		for j := k - 1; j > 0; j-- {
			w1 := float64(j)*lz - lgamma(float64(j+1)) - lgamma(-alp*float64(j))
			if w1 < w0-37 {
				break
			}
			ws += math.Exp(w1 - w0)
		}

		lpr -= math.Log(y)
		lpr += w0 + math.Log(ws)
	}

	return lpr
}

func (hmm *HMM) getPoissonLogObsProb(p, t, st int, exact bool) float64 {

	obs := hmm.Obs[p]

	if obs[t*hmm.NComp] == NullObs {
		panic("should not reach here")
	}

	var lpr float64
	ii := st * hmm.NComp
	for j := 0; j < hmm.NComp; j++ {
		y := obs[t*hmm.NComp+j]
		mn := hmm.Mean[ii]
		if mn < minPoissonMean {
			mn = minPoissonMean
		}
		lpr += -mn + y*math.Log(mn)
		ii++
	}

	if !exact {
		return lpr
	}

	for j := 0; j < hmm.NComp; j++ {
		y := obs[t*hmm.NComp+j]
		lpr -= lgamma(y + 1)
	}

	return lpr
}

func (hmm *HMM) getGaussianLogObsProb(p, t, st int, exact bool) float64 {

	obs := hmm.Obs[p]

	if obs[t*hmm.NComp] == NullObs {
		panic("should not reach here")
	}

	var lpr float64
	ii := st * hmm.NComp
	for j := 0; j < hmm.NComp; j++ {
		y := obs[t*hmm.NComp+j]
		mn := hmm.Mean[ii]
		sd := hmm.Std[ii]
		if hmm.ZeroInflated {
			zpr := hmm.Zprob[ii]
			if y == 0 {
				lpr += math.Log(zpr)
			} else {
				z := (y - mn) / sd
				q := -math.Log(sd) - z*z/2
				lpr += q + math.Log(1.0-zpr)
			}
		} else {
			z := (y - mn) / sd
			lpr += -math.Log(sd) - z*z/2
		}
		ii++
	}

	if !exact {
		return lpr
	}

	lpr -= float64(hmm.NState) * math.Log(2*math.Pi) / 2

	return lpr
}

// ForwardBackward calculates the forward and backward probabilities using
// the usual recursive approach.
func (hmm *HMM) ForwardBackward() {

	var wg sync.WaitGroup

	for p := 0; p < hmm.NParticle; p++ {
		wg.Add(1)
		go hmm.ForwardParticle(p, false, &wg)
		wg.Add(1)
		go hmm.BackwardParticle(p, &wg)
	}

	wg.Wait()
}

// Loglike returns the log-likelihood at the current parameter value.
func (hmm *HMM) Loglike() float64 {

	var wg sync.WaitGroup

	for p := 0; p < hmm.NParticle; p++ {
		wg.Add(1)
		go hmm.ForwardParticle(p, true, &wg)
	}

	wg.Wait()

	return floats.Sum(hmm.llf)
}

// AIC returns the AIC at the current parameter value.
func (hmm *HMM) AIC() float64 {

	df := 0
	df += hmm.NState - 1                // Initial state distribution
	df += hmm.NState * (hmm.NState - 1) // Transition matrix
	df += hmm.NState * hmm.NState       // Emission mean parameters

	if hmm.ObsModel == Gaussian {
		// Variance parameters
		switch hmm.VarModel {
		case VarFree:
			df += hmm.NState * hmm.NState
		case VarConst:
			df += 1
		case VarConstByState:
			df += hmm.NState
		case VarConstByComponent:
			df += hmm.NState
		default:
			panic("unkown varmodel")
		}

		// Zero inflation parameters
		if hmm.ZeroInflated {
			df += hmm.NState * hmm.NState
		}
	}

	return hmm.Loglike() - float64(df)
}

// ForwardParticle calculates the forward probabilities for one particle.
func (hmm *HMM) ForwardParticle(p int, exact bool, wg *sync.WaitGroup) {

	defer wg.Done()

	var llf float64
	fprob := hmm.Fprob[p]
	obs := hmm.Obs[p]

	// Due to concurrency, each particle needs its own workspace
	lt := make([]float64, hmm.NState*hmm.NState)
	terms := make([]float64, hmm.NState*hmm.NState)
	lfp := make([]float64, hmm.NState)

	// Precompute this
	for j := 0; j < hmm.NState*hmm.NState; j++ {
		lt[j] = math.Log(hmm.Trans[j])
	}

	j0 := -2 * hmm.NState // indexed of lagged observations
	j1 := -hmm.NState     // index of current observations
	k0 := -2 * hmm.NComp  // indexed of lagged observations
	k1 := -hmm.NComp      // index of current observations

	// Forward sweep
	for t := 0; t < hmm.NTime; t++ {

		j0 += hmm.NState
		j1 += hmm.NState
		k0 += hmm.NComp
		k1 += hmm.NComp

		if obs[k1] == NullObs {
			if k0 >= 0 && obs[k0] != NullObs {
				u := 0.0
				for j := 0; j < hmm.NState; j++ {
					u += fprob[j0+j]
				}
				llf += math.Log(u)
			}
			continue
		}

		// Initial time point
		if t == 0 || obs[k0] == NullObs {
			for j := 0; j < hmm.NState; j++ {
				fprob[j1+j] = math.Log(hmm.Init[j]) + hmm.GetLogObsProb(p, t, j, exact)
			}
			llf += hmm.normalizeMaxLog(fprob[j1 : j1+hmm.NState])
			continue
		}

		// Precompute this
		for st := 0; st < hmm.NState; st++ {
			lfp[st] = math.Log(fprob[j0+st])
		}

		// Calculate components of the update on the log scale.
		// Transition is from state st2 at time t-1 to state st1
		// at time t.
		for st1 := 0; st1 < hmm.NState; st1++ {
			yp := hmm.GetLogObsProb(p, t, st1, exact)
			for st2 := 0; st2 < hmm.NState; st2++ {
				terms[st1*hmm.NState+st2] = lfp[st2] + lt[st2*hmm.NState+st1] + yp
			}
		}

		// This shift does not change the result due to scale invariance
		mx := floats.Max(terms)
		llf += mx
		floats.AddConst(-mx, terms)

		// Get the probabilities by summing over possible histories.
		for st1 := 0; st1 < hmm.NState; st1++ {
			fprob[j1+st1] = 0
			for st2 := 0; st2 < hmm.NState; st2++ {
				fprob[j1+st1] += math.Exp(terms[st1*hmm.NState+st2])
			}
		}
		mx = hmm.normalizeMax(fprob[j1:j1+hmm.NState], 1)
		llf += math.Log(mx)
	}

	j0 = (hmm.NTime - 1) * hmm.NState
	k0 = (hmm.NTime - 1) * hmm.NComp
	if obs[k0] != NullObs {
		u := 0.0
		for j := 0; j < hmm.NState; j++ {
			u += fprob[j0+j]
		}
		llf += math.Log(u)
	}

	hmm.llf[p] = llf
}

// BackwardParticle calculates the backward probabilities for one particle.
func (hmm *HMM) BackwardParticle(p int, wg *sync.WaitGroup) {

	defer wg.Done()

	bprob := hmm.Bprob[p]
	obs := hmm.Obs[p]

	lt := make([]float64, hmm.NState*hmm.NState)
	lby := make([]float64, hmm.NState)
	terms := make([]float64, hmm.NState*hmm.NState)

	// Precompute this
	for j := 0; j < hmm.NState*hmm.NState; j++ {
		lt[j] = math.Log(hmm.Trans[j])
	}

	// Backward sweep
	j0 := hmm.NTime * hmm.NState
	j1 := hmm.NTime*hmm.NState + hmm.NState
	k0 := hmm.NTime * hmm.NComp
	k1 := hmm.NTime*hmm.NComp + hmm.NComp
	t := hmm.NTime
	for _i := 0; _i < hmm.NTime; _i++ {

		j0 -= hmm.NState
		j1 -= hmm.NState
		k0 -= hmm.NComp
		k1 -= hmm.NComp
		t -= 1

		if obs[k0] == NullObs {
			continue
		}

		// Initialize
		if t == hmm.NTime-1 || obs[k1] == NullObs {
			for st := 0; st < hmm.NState; st++ {
				bprob[j0+st] = 1
			}
			continue
		}

		for st := 0; st < hmm.NState; st++ {
			lby[st] = hmm.GetLogObsProb(p, t+1, st, false) + math.Log(bprob[j1+st])
		}

		// From st1 at t to st2 at t+1.
		for st1 := 0; st1 < hmm.NState; st1++ {
			for st2 := 0; st2 < hmm.NState; st2++ {
				terms[st1*hmm.NState+st2] = lby[st2] + lt[st1*hmm.NState+st2]
			}
		}

		floats.AddConst(-floats.Max(terms), terms)

		// Get the probabilities by summing over possible histories.
		for st1 := 0; st1 < hmm.NState; st1++ {
			bprob[j0+st1] = 0
			for st2 := 0; st2 < hmm.NState; st2++ {
				bprob[j0+st1] += math.Exp(terms[st1*hmm.NState+st2])
			}
		}

		hmm.normalizeMax(bprob[j0:j0+hmm.NState], 1)
	}
}

func (hmm *HMM) updateTransParticle(p int, newtrans, logtrans []float64, wg *sync.WaitGroup, mut *sync.Mutex) {

	defer wg.Done()
	obs := hmm.Obs[p]
	fprob := hmm.Fprob[p]
	bprob := hmm.Bprob[p]
	joint := make([]float64, hmm.NState*hmm.NState)
	jointsum := make([]float64, hmm.NState*hmm.NState)
	lcp := make([]float64, hmm.NState)

	for t := 0; t < hmm.NTime-1; t++ {

		if obs[t*hmm.NComp] == NullObs || obs[(t+1)*hmm.NComp] == NullObs {
			continue
		}

		for st := 0; st < hmm.NState; st++ {
			lcp[st] = hmm.GetLogObsProb(p, t+1, st, false) + math.Log(bprob[(t+1)*hmm.NState+st])
		}

		// Get the joint probabilities for states (st1, st2) on the log scale
		for st1 := 0; st1 < hmm.NState; st1++ {
			lfp := math.Log(fprob[t*hmm.NState+st1])
			for st2 := 0; st2 < hmm.NState; st2++ {
				joint[st1*hmm.NState+st2] = lfp + lcp[st2] + logtrans[st1*hmm.NState+st2]
			}
		}

		// Convert to probabilities
		floats.AddConst(-floats.Max(joint), joint)
		for j := range joint {
			joint[j] = math.Exp(joint[j])
		}
		normalizeSum(joint, 0)

		floats.Add(jointsum, joint)
	}

	mut.Lock()
	floats.Add(newtrans, jointsum)
	mut.Unlock()
}

// UpdateTrans updates the transmission probability matrix.
func (hmm *HMM) UpdateTrans() {

	newtrans := make([]float64, hmm.NState*hmm.NState)

	logtrans := make([]float64, hmm.NState*hmm.NState)
	for j := range hmm.Trans {
		logtrans[j] = math.Log(hmm.Trans[j])
	}

	var wg sync.WaitGroup
	var mut sync.Mutex
	for p := 0; p < hmm.NParticle; p++ {
		wg.Add(1)
		go hmm.updateTransParticle(p, newtrans, logtrans, &wg, &mut)
	}
	wg.Wait()

	// Normalize to probabilties by row
	for st := 0; st < hmm.NState; st++ {
		normalizeSum(newtrans[st*hmm.NState:(st+1)*hmm.NState], 1/float64(hmm.NState))
	}

	hmm.Trans = newtrans
}

// OracleTrans estimates the transmission probabilities from a known sequence of
// state values.
func (hmm *HMM) OracleTrans() []float64 {

	tr := make([]float64, hmm.NState*hmm.NState)

	for p := 0; p < hmm.NParticle; p++ {
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime-1; t++ {

			if obs[t*hmm.NComp] == NullObs || obs[(t+1)*hmm.NComp] == NullObs {
				continue
			}

			st1 := hmm.State[p][t]
			st2 := hmm.State[p][t+1]

			tr[st1*hmm.NState+st2]++
		}
	}

	for st := 0; st < hmm.NState; st++ {
		normalizeSum(tr[st*hmm.NState:(st+1)*hmm.NState], 1/float64(hmm.NState))
	}

	return tr
}

// OracleZprob estimates the probabilities that the observed value is zero using
// a known sequence of state values.
func (hmm *HMM) OracleZprob() []float64 {

	zpr := make([]float64, hmm.NState*hmm.NComp)
	den := make([]float64, hmm.NState)

	for p := 0; p < hmm.NParticle; p++ {
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime-1; t++ {

			if obs[t*hmm.NComp] == NullObs {
				continue
			}

			st1 := hmm.State[p][t]
			den[st1] += 1
			for j := 0; j < hmm.NComp; j++ {
				if obs[t*hmm.NComp+j] == 0 {
					zpr[st1*hmm.NComp+j] += 1
				}
			}
		}
	}

	for st := 0; st < hmm.NState; st++ {
		i := st * hmm.NComp
		floats.Scale(1/den[st], zpr[i:i+hmm.NComp])
	}

	return zpr
}

// MarginalMoments calculates the mean and standard deviation
// for each observed component, given that the value is not equal
// to zero.
func (hmm *HMM) MarginalMoments() ([]float64, []float64) {

	mean := make([]float64, hmm.NComp)
	num := make([]float64, hmm.NComp)
	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime; t++ {
			if obs[i] == NullObs {
				i += hmm.NComp
				continue
			}
			for j := 0; j < hmm.NComp; j++ {
				if obs[i] != 0 {
					mean[j] += obs[i]
					num[j]++
				}
				i++
			}
		}
	}
	floats.Div(mean, num)

	std := make([]float64, hmm.NComp)
	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime; t++ {
			if obs[t*hmm.NComp] == NullObs {
				i += hmm.NComp
				continue
			}
			for j := 0; j < hmm.NComp; j++ {
				y := obs[t*hmm.NComp+j]
				if !hmm.ZeroInflated || y != 0 {
					y -= mean[j]
					std[j] += y * y
				}
				i++
			}
		}
	}
	floats.Div(std, num)

	for i := range std {
		if math.IsNaN(mean[i]) || math.IsNaN(std[i]) {
			std[i] = 1
			mean[i] = 0
		}
		std[i] = math.Sqrt(std[i])
	}

	return mean, std
}

// OracleMoments estimates the means and standard deviations
// of the emissions given known state values.  This method
// can only be called if hmm.State is available.
func (hmm *HMM) OracleMoments() ([]float64, []float64) {

	mean := make([]float64, hmm.NState*hmm.NComp)
	std := make([]float64, hmm.NState*hmm.NComp)
	den := make([]float64, hmm.NState*hmm.NComp)

	for p := 0; p < hmm.NParticle; p++ {
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime; t++ {
			if obs[t*hmm.NComp] == NullObs {
				continue
			}
			st := hmm.State[p][t] // true state
			for j := 0; j < hmm.NComp; j++ {
				y := obs[t*hmm.NComp+j]
				if !hmm.ZeroInflated || y != 0 {
					mean[st*hmm.NComp+j] += y
					std[st*hmm.NComp+j] += y * y
					den[st*hmm.NComp+j] += 1
				}
			}
		}
	}

	for st := 0; st < hmm.NState; st++ {
		j := st * hmm.NComp
		for k := 0; k < hmm.NComp; k++ {
			mean[j] /= den[j]
			std[j] /= den[j]
			j++
		}
	}

	for st := 0; st < hmm.NState; st++ {
		j := st * hmm.NComp
		for k := 0; k < hmm.NComp; k++ {
			m := mean[j]
			std[j] -= m * m
			std[j] = math.Sqrt(std[j])
			j++
		}
	}

	return mean, std
}

func (hmm *HMM) WriteOracleSummary(labels []string) {

	hmm.parlogger.Printf("\nOracle statistics:\n")

	hmm.parlogger.Printf("Initial state distribution:\n")
	ist := hmm.OracleInit()
	hmm.writeMatrix(ist, 0, hmm.NState, 1, labels, nil)
	hmm.parlogger.Printf("\n")

	hmm.parlogger.Printf("Transition matrix:\n")
	tr := hmm.OracleTrans()
	hmm.writeMatrix(tr, 0, hmm.NState, hmm.NState, labels, labels)
	hmm.parlogger.Printf("\n")

	if hmm.ZeroInflated {
		hmm.parlogger.Printf("Zero probabilities:\n")
		zpr := hmm.OracleZprob()
		hmm.writeMatrix(zpr, 0, hmm.NState, hmm.NComp, labels, labels)
		hmm.parlogger.Printf("\n")
	}

	hmm.parlogger.Printf("Means:\n")
	mean, sd := hmm.OracleMoments()
	hmm.writeMatrix(mean, 0, hmm.NState, hmm.NComp, labels, labels)
	hmm.parlogger.Printf("\n")

	hmm.parlogger.Printf("Standard deviations:\n")
	hmm.writeMatrix(sd, 0, hmm.NState, hmm.NComp, labels, labels)
	hmm.parlogger.Printf("\n")
}

// OracleInit returns the frequency distribution of initial states using
// known state values.  This method can only be used if hmm.State is
// available.
func (hmm *HMM) OracleInit() []float64 {

	v := make([]float64, hmm.NState)

	for _, u := range hmm.State {
		// Use first non-null value for each particle
		for j := range u {
			if u[j] != NullState {
				v[u[j]]++
				break
			}
		}
	}

	normalizeSum(v, 1/float64(hmm.NState))
	return v
}

// UpdateInit updates the initial state probabilities.
func (hmm *HMM) UpdateInit() {

	vt := make([]float64, hmm.NState)
	zero(hmm.Init)

	for p := 0; p < hmm.NParticle; p++ {
		for t := 0; t < hmm.NTime; t++ {

			if hmm.Obs[p][t*hmm.NComp] == NullObs {
				continue
			}

			if t == 0 || hmm.Obs[p][(t-1)*hmm.NComp] == NullObs {
				fprob := hmm.Fprob[p][t*hmm.NState : (t+1)*hmm.NState]
				bprob := hmm.Bprob[p][t*hmm.NState : (t+1)*hmm.NState]

				floats.MulTo(vt, fprob, bprob)
				normalizeSum(vt, 0)
				floats.Add(hmm.Init, vt)
			}
		}
	}

	normalizeSum(hmm.Init, 1/float64(hmm.NState))
}

// UpdateObsParams updates the parameters of the emission model.
func (hmm *HMM) UpdateObsParams() {

	switch hmm.ObsModel {
	case Gaussian:
		hmm.updateGaussianObsParams()
	case Poisson:
		hmm.updatePoissonObsParams()
	case Tweedie:
		// Can use the same update as Poisson
		hmm.updatePoissonObsParams()
	default:
		panic("Unkown observation model")
	}
}

func (hmm *HMM) updateGaussianObsParams() {

	pr := make([]float64, hmm.NState)
	pt := make([]float64, hmm.NState)

	zero(hmm.Mean)
	zero(hmm.Zprob)

	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		obs := hmm.Obs[p]
		fprob := hmm.Fprob[p]
		bprob := hmm.Bprob[p]

		for t := 0; t < hmm.NTime; t++ {

			if obs[t*hmm.NComp] != NullObs {
				floats.MulTo(pr, fprob[i:i+hmm.NState], bprob[i:i+hmm.NState])
				normalizeSum(pr, 0)
				floats.Add(pt, pr)

				for st := 0; st < hmm.NState; st++ {
					for j := 0; j < hmm.NComp; j++ {
						y := obs[t*hmm.NComp+j]
						if hmm.ZeroInflated {
							if y == 0.0 {
								hmm.Zprob[st*hmm.NComp+j] += pr[st]
							} else {
								hmm.Mean[st*hmm.NComp+j] += pr[st] * y
							}
						} else {
							hmm.Mean[st*hmm.NComp+j] += pr[st] * y
						}
					}
				}
			}

			i += hmm.NState
		}
	}

	for st := 0; st < hmm.NState; st++ {
		i, j := st*hmm.NComp, (st+1)*hmm.NComp
		var s = 1 / pt[st]
		if pt[st] < 1e-10 {
			hmm.msglogger.Printf("Underflow in Gaussian SD update\n")
			s = 0
		}
		if hmm.ZeroInflated {
			floats.Scale(s, hmm.Zprob[i:j])
		}
		floats.Scale(s, hmm.Mean[i:j])
	}

	if hmm.ZeroInflated {
		// Truncate the zero-probabilities
		for i := range hmm.Zprob {
			if hmm.Zprob[i] > zpmax {
				hmm.Zprob[i] = zpmax
			}
		}

		for i := 0; i < hmm.NState*hmm.NComp; i++ {
			hmm.Mean[i] /= (1.0 - hmm.Zprob[i])
		}
	}
}

func (hmm *HMM) updatePoissonObsParams() {

	pr := make([]float64, hmm.NState)
	pt := make([]float64, hmm.NState)

	zero(hmm.Mean)

	for p := 0; p < hmm.NParticle; p++ {
		i := -hmm.NState
		k := -hmm.NComp
		obs := hmm.Obs[p]
		fprob := hmm.Fprob[p]
		bprob := hmm.Bprob[p]

		for t := 0; t < hmm.NTime; t++ {

			i += hmm.NState
			k += hmm.NComp

			if obs[k] == NullObs {
				continue
			}

			floats.MulTo(pr, fprob[i:i+hmm.NState], bprob[i:i+hmm.NState])
			normalizeSum(pr, 0)
			floats.Add(pt, pr)
			for st := 0; st < hmm.NState; st++ {
				for j := 0; j < hmm.NComp; j++ {
					y := obs[t*hmm.NComp+j]
					hmm.Mean[st*hmm.NComp+j] += pr[st] * y
				}
			}
		}
	}

	for st := 0; st < hmm.NState; st++ {
		i, j := st*hmm.NComp, (st+1)*hmm.NComp
		var s = 1 / pt[st]
		if pt[st] < 1e-10 {
			hmm.msglogger.Printf("Underflow in Poisson update...")
			s = 0
		}
		floats.Scale(s, hmm.Mean[i:j])
	}
}

func (hmm *HMM) UpdateVarFree() {

	pr := make([]float64, hmm.NState)
	pt := make([]float64, hmm.NState)

	zero(hmm.Std)

	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		k := 0
		obs := hmm.Obs[p]
		fprob := hmm.Fprob[p]
		bprob := hmm.Bprob[p]

		for t := 0; t < hmm.NTime; t++ {

			floats.MulTo(pr, fprob[i:i+hmm.NState], bprob[i:i+hmm.NState])
			normalizeSum(pr, 0)
			floats.Add(pt, pr)

			if obs[t*hmm.NComp] != NullObs {
				for st := 0; st < hmm.NState; st++ {
					for j := 0; j < hmm.NComp; j++ {
						y := obs[t*hmm.NComp+j]
						if !hmm.ZeroInflated || y != 0.0 {
							y -= hmm.Mean[st*hmm.NComp+j]
							hmm.Std[st*hmm.NComp+j] += pr[st] * y * y
						}
					}
				}
			}

			i += hmm.NState
			k += hmm.NComp
		}
	}

	for st := 0; st < hmm.NState; st++ {
		i, j := st*hmm.NComp, (st+1)*hmm.NComp
		var s = 1 / pt[st]
		if pt[st] < 1e-10 {
			print("Underflow in Std update")
			s = 0
		}
		floats.Scale(s, hmm.Std[i:j])
	}

	for i := 0; i < hmm.NState*hmm.NComp; i++ {
		if hmm.ZeroInflated {
			hmm.Std[i] = math.Sqrt(hmm.Std[i] / (1.0 - hmm.Zprob[i]))
		} else {
			hmm.Std[i] = math.Sqrt(hmm.Std[i])
		}
		if hmm.Std[i] < sdmin {
			hmm.Std[i] = sdmin
			hmm.Warnings.SDTruncate++
		}
	}
}

func (hmm *HMM) UpdateVarConstByState() {

	pr := make([]float64, hmm.NState)
	pt := make([]float64, hmm.NState)

	zero(hmm.Std)

	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		obs := hmm.Obs[p]
		fprob := hmm.Fprob[p]
		bprob := hmm.Bprob[p]
		for t := 0; t < hmm.NTime; t++ {

			if obs[t*hmm.NComp] != NullObs {

				floats.MulTo(pr, fprob[i:i+hmm.NState], bprob[i:i+hmm.NState])
				normalizeSum(pr, 0)

				for st := 0; st < hmm.NState; st++ {
					for j := 0; j < hmm.NComp; j++ {
						y := obs[t*hmm.NComp+j]
						if !hmm.ZeroInflated || y != 0.0 {
							y -= hmm.Mean[st*hmm.NComp+j]
							hmm.Std[j] += pr[st] * y * y
							pt[j] += pr[st]
						}
					}
				}
			}

			i += hmm.NState
		}
	}

	for i := 0; i < hmm.NComp; i++ {
		hmm.Std[i] /= pt[i]
	}

	if hmm.ZeroInflated {
		for i := range hmm.Std {
			hmm.Std[i] = math.Sqrt(hmm.Std[i] / (1.0 - hmm.Zprob[i]))
		}
	} else {
		for i := range hmm.Std {
			hmm.Std[i] = math.Sqrt(hmm.Std[i])
		}
	}

	// Cap the SD parameters
	for i := range hmm.Std {
		if hmm.Std[i] < sdmin {
			hmm.Std[i] = sdmin
			hmm.Warnings.SDTruncate++
		}
	}

	// The remaining rows are constrained to be equal to the first row.
	for i := 1; i < hmm.NState; i++ {
		j := i * hmm.NComp
		copy(hmm.Std[j:j+hmm.NComp], hmm.Std[0:hmm.NComp])
	}
}

func (hmm *HMM) UpdateVarConstByComponent() {

	pr := make([]float64, hmm.NState)
	pt := make([]float64, hmm.NState)
	zero(hmm.Std)

	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		obs := hmm.Obs[p]
		fprob := hmm.Fprob[p]
		bprob := hmm.Bprob[p]
		for t := 0; t < hmm.NTime; t++ {

			if obs[t*hmm.NComp] != NullObs {
				floats.MulTo(pr, fprob[i:i+hmm.NState], bprob[i:i+hmm.NState])
				normalizeSum(pr, 0)

				for j := 0; j < hmm.NComp; j++ {
					y := obs[t*hmm.NComp+j]
					if !hmm.ZeroInflated || y != 0.0 {
						for st := 0; st < hmm.NState; st++ {
							r := y - hmm.Mean[st*hmm.NComp+j]
							hmm.Std[st*hmm.NComp] += pr[st] * r * r
							pt[st] += pr[st]
						}
					}
				}
			}

			i += hmm.NState
		}
	}

	for st := 0; st < hmm.NState; st++ {
		hmm.Std[st*hmm.NComp] /= pt[st]
	}

	for st := 0; st < hmm.NState; st++ {
		for j := 1; j < hmm.NComp; j++ {
			hmm.Std[st*hmm.NComp+j] = hmm.Std[st*hmm.NComp]
		}
	}

	if hmm.ZeroInflated {
		for i := range hmm.Std {
			hmm.Std[i] = math.Sqrt(hmm.Std[i] / (1.0 - hmm.Zprob[i]))
		}
	} else {
		for i := range hmm.Std {
			hmm.Std[i] = math.Sqrt(hmm.Std[i])
		}
	}

	// Cap the SD parameters
	for i := range hmm.Std {
		if hmm.Std[i] < sdmin {
			hmm.Std[i] = sdmin
			hmm.Warnings.SDTruncate++
		}
	}
}

func (hmm *HMM) UpdateVarConst() {

	pr := make([]float64, hmm.NState)
	zero(hmm.Std)
	var pt, vp float64

	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		obs := hmm.Obs[p]
		fprob := hmm.Fprob[p]
		bprob := hmm.Bprob[p]
		for t := 0; t < hmm.NTime; t++ {

			if obs[t*hmm.NComp] != NullObs {

				floats.MulTo(pr, fprob[i:i+hmm.NState], bprob[i:i+hmm.NState])
				normalizeSum(pr, 0)

				for st := 0; st < hmm.NState; st++ {
					for j := 0; j < hmm.NComp; j++ {
						y := obs[t*hmm.NComp+j]
						if !hmm.ZeroInflated || y != 0.0 {
							y -= hmm.Mean[st*hmm.NComp+j]
							vp += pr[st] * y * y
							pt += pr[st]
						}
					}
				}
			}

			i += hmm.NState
		}
	}

	for i := range hmm.Std {
		hmm.Std[i] = vp / pt
	}

	if hmm.ZeroInflated {
		for i := range hmm.Std {
			hmm.Std[i] = math.Sqrt(hmm.Std[i] / (1.0 - hmm.Zprob[i]))
		}
	} else {
		for i := range hmm.Std {
			hmm.Std[i] = math.Sqrt(hmm.Std[i])
		}
	}

	// Truncate the SD parameters
	for i := range hmm.Std {
		if hmm.Std[i] < sdmin {
			hmm.Std[i] = sdmin
			hmm.Warnings.SDTruncate++
		}
	}
}

// Fit uses the EM algorithm to estimate the structural parameters of the HMM.
func (hmm *HMM) Fit(maxiter int) {

	hmm.LLF = make([]float64, 0, maxiter)

	hmm.msglogger.Printf("Estimating model parameters...\n")
	var llf float64

	for i := 0; i < maxiter; i++ {
		hmm.msglogger.Printf("Beginning ForwardBackward...")
		hmm.ForwardBackward()
		hmm.msglogger.Printf("Beginning UpdateTrans...")
		hmm.UpdateTrans()
		hmm.msglogger.Printf("Beginning UpdateInit...")
		hmm.UpdateInit()
		hmm.msglogger.Printf("Beginning UpdateObsParams...")
		hmm.UpdateObsParams()

		if hmm.ObsModel == Gaussian {
			hmm.msglogger.Printf("Updating variance parameters...")
			switch hmm.VarModel {
			case VarFree:
				hmm.UpdateVarFree()
			case VarConst:
				hmm.UpdateVarConst()
			case VarConstByState:
				hmm.UpdateVarConstByState()
			case VarConstByComponent:
				hmm.UpdateVarConstByComponent()
			default:
				panic("not implemented")
			}
		}

		llfnew := floats.Sum(hmm.llf)
		if i > 0 {
			if llfnew < llf-1e-10 {
				hmm.msglogger.Printf("Log-likelihood decreased by %f\n", llf-llfnew)
				hmm.Warnings.LogLikeDecreased++
			} else if llfnew-llf < 1e-8 {
				hmm.msglogger.Printf("Converged at iteration %d\n", i)
				break
			}
		}

		llf = llfnew
		hmm.LLF = append(hmm.LLF, llf)
		hmm.msglogger.Printf("llf=%f\n", llf)
	}

	hmm.msglogger.Printf("%+v\n", hmm.Warnings)
}

// WriteSummary writes the model parameters to the parameter logger.
// The optional row/column labels are used if provided.
func (hmm *HMM) WriteSummary(labels []string, title string) {

	hmm.parlogger.Printf(title)
	hmm.parlogger.Printf("\n")

	hmm.parlogger.Printf("Initial states distribution:\n")
	hmm.writeMatrix(hmm.Init, 0, hmm.NState, 1, labels, nil)
	hmm.parlogger.Printf("\n")

	hmm.parlogger.Printf("Transition matrix:\n")
	hmm.writeMatrix(hmm.Trans, 0, hmm.NState, hmm.NState, labels, labels)
	hmm.parlogger.Printf("\n")

	if hmm.ZeroInflated {
		hmm.parlogger.Printf("Zero probabilties:\n")
		hmm.writeMatrix(hmm.Zprob, 0, hmm.NState, hmm.NComp, labels, labels)
		hmm.parlogger.Printf("\n")
	}

	hmm.parlogger.Printf("Means:\n")
	hmm.writeMatrix(hmm.Mean, 0, hmm.NState, hmm.NComp, labels, labels)
	hmm.parlogger.Printf("\n")

	if hmm.ObsModel == Gaussian {
		hmm.parlogger.Printf("Standard deviations:\n")
		hmm.writeMatrix(hmm.Std, 0, hmm.NState, hmm.NComp, labels, labels)
		hmm.parlogger.Printf("\n")
	}
}

// writeMatrix writes a matrix in text format to the logger
func (hmm *HMM) writeMatrix(x []float64, off, nrow, ncol int, rowlabels, collabels []string) {

	var buf bytes.Buffer

	if rowlabels != nil && nrow != len(rowlabels) {
		msg := "len(rowlabels) != nrow\n"
		_, _ = io.WriteString(os.Stderr, msg)
	}

	if collabels != nil {
		if ncol != len(collabels) {
			msg := "len(collabels) != ncol\n"
			_, _ = io.WriteString(os.Stderr, msg)
		}
		if rowlabels != nil {
			_, _ = io.WriteString(&buf, fmt.Sprintf("%20s", ""))
		}
		for _, c := range collabels {
			_, _ = io.WriteString(&buf, fmt.Sprintf("%20s", c))
		}
		hmm.parlogger.Printf(buf.String())
	}

	for i := 0; i < nrow; i++ {

		buf.Reset()

		if rowlabels != nil {
			_, _ = io.WriteString(&buf, fmt.Sprintf("%-20s", rowlabels[i]))
		}
		for j := 0; j < ncol; j++ {
			_, _ = io.WriteString(&buf, fmt.Sprintf("%20.4f", x[off+i*ncol+j]))
		}

		hmm.parlogger.Printf(buf.String())
	}
}

// normalize the values in x from index i to index i+q to have a maxiumum of 1.
func (hmm *HMM) normalizeMax(x []float64, z float64) float64 {
	mx := floats.Max(x)
	if mx < 1e-15 {
		panic(fmt.Sprintf("normalizemax: %v\n", x))
		hmm.Warnings.NormalizeMaxThreshold++
		for j := range x {
			x[j] = z
		}
		return 0
	}
	floats.Scale(1/mx, x)
	return mx
}

// Subtract the maximum value from x, then exponentiate.
func (hmm *HMM) normalizeMaxLog(x []float64) float64 {
	mx := floats.Max(x)
	floats.AddConst(-mx, x)
	for j := range x {
		x[j] = math.Exp(x[j])
	}

	return mx
}

// normalize the values in x from index i to index i+q to have a sum of 1.
func normalizeSum(x []float64, z float64) {
	scale := floats.Sum(x)
	if scale < 1e-10 {
		for j := range x {
			x[j] = z
		}
		return
	}
	floats.Scale(1/scale, x)
}

func argmax(x []float64) int {
	j := 0
	v := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > v {
			v = x[i]
			j = i
		}
	}

	return j
}

// Zero the elements of x
func zero(x []float64) {
	for j := range x {
		x[j] = 0
	}
}

// makeIntArray makes a collection of r slices
// of length c, packed contiguously.
func makeIntArray(r, c int) [][]int {

	bka := make([]int, r*c)
	x := make([][]int, r)
	ii := 0
	for j := 0; j < r; j++ {
		x[j] = bka[ii : ii+c]
		ii += c
	}

	return x
}

// makeFloatArray makes a collection of r slices
// of length c, packed contiguously.
func makeFloatArray(r, c int) [][]float64 {

	bka := make([]float64, r*c)
	x := make([][]float64, r)
	ii := 0
	for j := 0; j < r; j++ {
		x[j] = bka[ii : ii+c]
		ii += c
	}

	return x
}

// ReadHMM reads a compressed HMM value from a gzip-compressed
// gob file.
func ReadHMM(fname string) *HMM {

	fid, err := os.Open(fname)
	if err != nil {
		panic(err)
	}
	defer fid.Close()

	gid, err := gzip.NewReader(fid)
	if err != nil {
		panic(err)
	}
	defer gid.Close()

	dec := gob.NewDecoder(gid)

	var hmm HMM
	if err := dec.Decode(&hmm); err != nil {
		panic(err)
	}

	return &hmm
}

// CompareStates returns the number of positions where the state
// sequences x and y agree, and the number of positions in which
// x and y are not null.  Panics if the lengths of x and y differ,
// or if there is any position in which exactly one of x and y is
// null.
func CompareStates(x, y []int) (int, int) {

	if len(x) != len(y) {
		panic("Lengths are not equal")
	}

	var e, n int
	for t := range x {
		if x[t] == NullState && y[t] != NullState {
			panic("inconsistent")
		}
		if x[t] != NullState && y[t] == NullState {
			panic("inconsistent")
		}
		if x[t] == NullState {
			continue
		}
		if x[t] != y[t] {
			e++
		}
		n++
	}

	return e, n
}
