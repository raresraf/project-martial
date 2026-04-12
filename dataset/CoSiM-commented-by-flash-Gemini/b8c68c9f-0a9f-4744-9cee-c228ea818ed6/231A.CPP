/**
 * @file 231A.CPP
 * @brief Logic for consensus-based filtering in a multi-agent decision problem.
 * 
 * Logic: Calculates the total number of problems a team will attempt. A problem 
 * is chosen if at least two out of three team members are confident (state = 1) 
 * in its solution.
 * 
 * Algorithm: Two-pass processing (population then threshold-based reduction).
 * Time Complexity: $O(N)$ where N is the number of problems.
 * Space Complexity: $O(N)$ for storing all problem-agent states in a 2D matrix.
 */

#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  int n;
  // Functional Utility: Reads the total headcount of problems to evaluate.
  if (!(cin >> n)) return 0;

  // Data Structure: Matrix to buffer confidence levels for N problems across 3 agents.
  int arr[n][3];
  
  /**
   * Block Logic: Population phase.
   * Invariant: Fills the matrix with binary confidence values from standard input.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }

  int res = 0;
  /**
   * Block Logic: Reduction phase (solvability evaluation).
   * Invariant: Evaluates each problem against the majority consensus threshold (>= 2).
   */
  for (int i = 0; i < n; i++) {
    int counter = 0;
    // Functional Utility: Aggregates individual agent confidence for problem 'i'.
    for (int j = 0; j < 3; j++) {
      if (arr[i][j] == 1)
        counter++;
    }
    
    // Logic: A consensus of 2 or more agents increments the solvable problem tally.
    if (counter >= 2) {
      res++;
    }
  }

  // Output: Total count of problems satisfying the consensus rule.
  cout << res;
  
  return 0;
}
