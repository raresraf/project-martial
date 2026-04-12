/**
 * @file 231A.CPP
 * @brief Logic for consensus-based decision-making in a multi-agent environment.
 * 
 * Algorithm: Majority voting filter (2 out of 3).
 * Time Complexity: $O(N)$ where N is the number of problems.
 * Space Complexity: $O(1)$ as it uses constant memory for state tracking.
 */

#include <iostream>

using std::cin;
using std::cout;
using std::endl;

int main() {
  // Functional Utility: Reads the total number of problems to be evaluated.
  int n;
  cin >> n;
  
  // States of the three agents (Petya, Vasya, Tonya) regarding a problem.
  int a, b, c;
  int problemSolved = 0;

  /**
   * Block Logic: Iteratively evaluates problem feasibility based on agent consensus.
   * Invariant: A problem is counted as solvable if and only if >= 2 agents 
   * confirm their confidence.
   */
  for (int i = 0; i != n; ++i) {
    // Reads confidence levels (0 or 1) for each agent.
    cin >> a >> b >> c;
    
    // Logic: Applies a threshold-based summation to determine majority confidence.
    if ((a + b + c) >= 2) {
      problemSolved++;
    }
  }
  
  // Output the total count of solvable problems based on the consensus rule.
  cout << problemSolved;
  
  return 0;
}
