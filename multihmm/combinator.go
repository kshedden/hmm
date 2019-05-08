package hmmlib

import (
	"encoding/binary"
	"hash"
	"hash/fnv"
	"math/rand"
)

type combinator struct {

	// scores[p] contains an ascending sequence of numbers for particle p
	// that quantify how well placing particle p in a given state fits the
	// data.  Internally to this value, scores[p][j] is the score for
	// placing particle p in state j, but externally to this class, state
	// j is mapped to a different actual state.
	scores [][]float64

	// The joint state must satisfy this constraint.  If nil, no constraint
	// is imposed.
	constraint ConstraintFunc

	// hash is used for quickly determining if two slices are equal
	hash hash.Hash64

	// mask[p] is true indicates that particle p should be omitted from
	// all calculations
	mask []bool

	// seen[hashix[x]] is true iff the slic x has already been included in
	// the enumeration
	seen map[uint64]bool

	// Workspace for project
	prjwk []int
	ixwk  []int
}

// combiRec represents a single set of state assignments with its score
type combiRec struct {
	score float64
	ix    []int
}

// hashix creates an integer hash value from the given slice.
func (combi *combinator) hashix(ix []int) uint64 {

	combi.hash.Reset()
	for p, i := range ix {
		if !combi.mask[p] {
			err := binary.Write(combi.hash, binary.LittleEndian, uint64(i))
			if err != nil {
				panic(err)
			}
		}
	}

	return combi.hash.Sum64()
}

// getScore returns the negative likelihood score for the given set of
// assignments.  Smaller scores correspond to greater likelihood.
func (combi *combinator) getScore(ix []int) float64 {

	var v float64
	for p, j := range ix {
		if !combi.mask[p] {
			v += combi.scores[p][j]
		}
	}

	return v
}

// NewCombinator returns a newly allocated combinator object, for the given parameters.
func NewCombinator(obspr [][]float64, constraint ConstraintFunc, mask []bool) *combinator {

	return &combinator{
		scores:     obspr,
		constraint: constraint,
		hash:       fnv.New64(),
		mask:       mask,
		prjwk:      make([]int, 0, len(mask)),
		ixwk:       make([]int, len(mask)),
	}
}

// copy creates a copy of the slice
func (combi *combinator) copy(ix []int) []int {
	ixc := make([]int, len(ix))
	copy(ixc, ix)
	return ixc
}

// project takes a slice and randomly projects it to
// a nearby slice that satisfies the constraint.
func (combi *combinator) project(ix, caps []int) bool {

	nstate := len(combi.scores[0])

	copy(combi.ixwk, ix)

	// Get indices of non-masked particles
	prjwk := combi.prjwk[0:0]
	for i, m := range combi.mask {
		if !m {
			prjwk = append(prjwk, i)
		}
	}

	for iter := 0; iter < 5000; iter++ {

		// Make a random move
		q := prjwk[rand.Int()%len(prjwk)]

		if ix[q] < caps[q] {
			ix[q] = caps[q]
		} else {
			ix[q]++
		}

		if ix[q] >= nstate {
			ix[q] = combi.ixwk[q]
		}

		if combi.constraint(ix, combi.mask) > 0 {
			continue
		}

		ha := combi.hashix(ix)
		if !combi.seen[ha] {
			combi.seen[ha] = true
			return true
		}
	}

	return false
}

// enumerate produces an array of distinct states with relatively high
// likelihood.  The states are sorted in decreasing likelihood order.
func (combi *combinator) enumerate(caps []int) []combiRec {

	// Check validity of caps
	for j := range caps {
		if !combi.mask[j] && caps[j] < 1 {
			panic("Invalid caps")
		}
	}

	// Use this to avoid including the same state multiple times
	combi.seen = make(map[uint64]bool)

	var rv []combiRec
	state := make([]int, len(caps))

	for {
		// Handle the current state vector
		ix := combi.copy(state)
		if combi.constraint(state, combi.mask) == 0 {
			// The constraint is satisfied
			rv = append(rv, combiRec{combi.getScore(ix), ix})
			h := combi.hashix(ix)
			combi.seen[h] = true
		} else {
			// The constraint is not satisfied
			if combi.project(ix, caps) {
				rv = append(rv, combiRec{combi.getScore(ix), ix})
			}
		}

		// Advance the state
		for j := range state {
			if !combi.mask[j] && state[j] < caps[j]-1 {
				state[j]++
				break
			}
			state[j] = 0
		}

		done := true
		for j := range state {
			if !combi.mask[j] && state[j] != 0 {
				done = false
			}
		}
		if done {
			break
		}
	}

	return rv
}
