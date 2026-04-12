/**
 * @file 231A.CPP
 * @brief Logic for consensus-based decision filtering in a multi-agent team.
 * 
 * Logic: Calculates the total number of problems solvable by a team of three agents. 
 * A problem is considered solvable if at least two out of three agents are 
 * confident (state = 1).
 * 
 * Algorithm: Explicit pairwise consensus checking (2-out-of-3 logic).
 * Time Complexity: $O(N)$ where N is the number of problems.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <iostream>

using namespace std;

int main() {
  int n = 0, x = 0, y = 0, z = 0, solvable_count = 0;
  
  // Functional Utility: Reads total problem count.
  if (!(cin >> n)) return 0;

  /**
   * Block Logic: Iteratively evaluates problem feasibility based on agent consensus.
   * Invariant: Increments solvable_count if any pair of agents is confident.
   */
  for (int i = 0; i < n; i++) {
    // Reads confidence states for agents Petya (x), Vasya (y), and Tonya (z).
    cin >> x >> y >> z;
    
    /**
     * Functional Utility: Pairwise majority evaluation.
     * Logic: Exhaustive check of all 2-agent combinations to determine if threshold is met.
     */
    if (x == 1 && y == 1) {
      solvable_count++;
    } else if (x == 1 && z == 1) {
      solvable_count++;
    } else if (y == 1 && z == 1) {
      solvable_count++;
    }
  }
  
  // Output the total count of solvable problems.
  cout << solvable_count << endl;
  
  return 0;
}
