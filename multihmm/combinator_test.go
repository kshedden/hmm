package hmmlib

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

func Test1(t *testing.T) {

	for pix, p := range []struct {
		scores     [][]float64
		caps       []int
		constraint func([]int, []bool) float64
		expected   []combiRec
	}{
		{
			scores: [][]float64{
				{1, 3, 4},
				{2, 7, 11},
				{5, 9, 10},
			},
			caps: []int{2, 3, 2},
			constraint: func(ix []int, mask []bool) float64 {
				// Count number of repeated values
				u := make([]int, len(ix))
				copy(u, ix)
				sort.IntSlice(u).Sort()
				m := 0.0
				for j := 1; j < len(u); j++ {
					if u[j] == u[j-1] {
						m++
					}
				}
				return m
			},
			expected: []combiRec{
				{15, []int{1, 0, 2}},
				{18, []int{0, 1, 2}},
				{16, []int{2, 1, 0}},
				{19, []int{1, 2, 0}},
				{15, []int{2, 0, 1}},
				{21, []int{0, 2, 1}},
			},
		},
		{
			scores: [][]float64{
				{1, 3},
				{2, 7},
				{5, 9},
			},
			caps: []int{1, 2, 1},
			constraint: func(ix []int, mask []bool) float64 {
				return 0
			},
			expected: []combiRec{
				{8, []int{0, 0, 0}},
				{13, []int{0, 1, 0}},
			},
		},
		{
			scores: [][]float64{
				{1, 3},
				{2, 7},
				{5, 9},
			},
			caps: []int{1, 1, 2},
			constraint: func(ix []int, mask []bool) float64 {
				return 0
			},
			expected: []combiRec{
				{8, []int{0, 0, 0}},
				{12, []int{0, 0, 1}},
			},
		},
	} {
		combi := NewCombinator(p.scores, p.constraint, make([]bool, len(p.scores)))

		rand.Seed(6)
		x := combi.enumerate(p.caps)

		for i := range x {
			if x[i].score != p.expected[i].score {
				fmt.Printf("[A] pix=%d, i=%d\n", pix, i)
				fmt.Printf("observed=%v\n", x)
				fmt.Printf("expected=%v\n", p.expected)
				t.Fail()
			}
			if !intSliceEqual(x[i].ix, p.expected[i].ix) {
				fmt.Printf("[B] pix=%d, i=%d\n", pix, i)
				fmt.Printf("observed=%v\n", x)
				fmt.Printf("expected=%v\n", p.expected)
				t.Fail()
			}
		}
	}
}

func intSliceEqual(u, v []int) bool {

	if len(u) != len(v) {
		return false
	}

	for i := range u {
		if u[i] != v[i] {
			return false
		}
	}

	return true
}
