/**
 * @file 231A.CPP
 * @brief Logic for consensus-based filtering in a multi-agent decision task.
 * 
 * Logic: Calculates the number of problems a team will attempt. A problem 
 * is selected if a majority (>= 2) of the three team members are confident 
 * in their solution.
 * 
 * Algorithm: Per-problem threshold-based summation.
 * Time Complexity: $O(N)$ where N is the number of problems.
 * Space Complexity: $O(1)$ auxiliary space beyond the small input buffer.
 */

#include <iostream>

using namespace std;

int main() {
  int n, arr[1000], count = 0, current_problem_total = 0;
  
  // Functional Utility: Reads total problem count.
  if (!(cin >> n)) return 0;

  /**
   * Block Logic: Main problem evaluation loop.
   * Invariant: Evaluates each problem independently.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Block Logic: Agent confidence aggregation.
     * Logic: Sums the binary confidence states (0 or 1) of three team members.
     * Note: Inner loop reuses variable 'i', shadowing the outer scope.
     */
    for (int i = 0; i < 3; i++) {
      cin >> arr[i];
      current_problem_total += arr[i];
    }

    /**
     * Functional Utility: Majority rule verification.
     * Invariant: A problem is solvable if confidence sum >= 2.
     */
    if (current_problem_total >= 2)
      count += 1;
    
    // State Reset: Clears accumulator for the next problem in the sequence.
    current_problem_total = 0;
  }

  // Output the total number of problems meeting the consensus threshold.
  cout << count << endl;

  return 0;
}
