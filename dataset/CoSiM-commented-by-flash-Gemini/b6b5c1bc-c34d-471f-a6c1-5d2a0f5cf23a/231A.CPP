/**
 * @file 231A.CPP
 * @brief Logic for consensus-based filtering in a multi-agent decision task.
 * 
 * Logic: Calculates the number of problems a team of three agents will attempt. 
 * A problem is selected only if a majority (at least 2 out of 3) of the agents 
 * express confidence (state = 1) in the solution.
 * 
 * Algorithm: Majority voting threshold (sum >= 2).
 * Time Complexity: $O(N)$ where N is the number of problems.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  int n;
  // Functional Utility: Reads the total number of problems to be evaluated.
  if (!(cin >> n)) return 0;

  // State: Accumulator for the count of solvable problems.
  int counter = 0;

  /**
   * Block Logic: Iteratively evaluates problem solvability based on agent consensus.
   * Invariant: A problem is counted as solvable if and only if >= 2 agents 
   * confirm their confidence.
   */
  for (int i = 0; i < n; i++) {
    int a, b, c;
    // Reads binary confidence states for each agent.
    cin >> a >> b >> c;
    
    /**
     * Functional Utility: Majority rule verification.
     * Logic: Sum of states {0, 1} >= 2 correctly identifies the consensus threshold.
     */
    if ((a + b + c) >= 2)
      counter++;
  }

  // Output: Total count of solvable problems.
  cout << counter;
  
  return 0;
}
