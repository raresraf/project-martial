/**
 * @file 231A.CPP
 * @brief Logic for consensus-based decision-making in a multi-agent team.
 * 
 * Logic: Calculates the total number of problems solvable by a team of three agents. 
 * A problem is considered solvable if at least two out of three agents are 
 * confident (state = 1) in the solution.
 * 
 * Algorithm: Majority voting filter (threshold = 2).
 * Time Complexity: $O(N)$ where N is the number of problems.
 * Space Complexity: $O(N)$ for storing all problem states in a 2D matrix.
 */

#include <iostream>
using namespace std;

int main() {
  // Data Structure: Matrix to buffer confidence levels for N problems across 3 agents.
  int arr[1000][3];
  int n;
  cin >> n;

  // Block Logic: Populates the problem-agent state matrix.
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= 2; j++) {
      cin >> arr[i][j];
    }
  }

  int result = 0;

  /**
   * Block Logic: Evaluates each problem for solvability based on consensus.
   * Invariant: A problem is solvable if and only if sum(states) >= 2.
   */
  for (int i = 0; i < n; i++) {
    int confidence_count = 0;
    for (int j = 0; j <= 2; j++) {
      // Logic: Increments count for each confident agent.
      if (arr[i][j] == 1)
        confidence_count++;
      
      /**
       * Functional Utility: Early exit for the inner loop once the threshold 
       * of 2 agents is reached.
       */
      if (confidence_count == 2) {
        result++;
        break;
      }
    }
  }
  
  // Output the total count of solvable problems.
  cout << result;
  
  return 0;
}
