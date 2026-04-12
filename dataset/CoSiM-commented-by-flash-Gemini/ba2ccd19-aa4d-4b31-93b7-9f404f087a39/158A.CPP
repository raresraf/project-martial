/**
 * @file 158A.CPP
 * @brief Logic for determining competition qualification benchmarks.
 * 
 * Logic: Calculates the count of participants qualifying for the next stage. 
 * Progression usually requires a score >= k-th rank and score > 0.
 * 
 * Algorithm: Single-pass scan with piecewise conditional logic for early termination.
 * Time Complexity: $O(N)$ where N is the participant count.
 * Space Complexity: $O(N)$ for participant score storage in a vector.
 */

#include <iostream>
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k;
  // Functional Utility: Reads total participants (n) and the target rank (k).
  if (!(cin >> n >> k)) return 0;
  
  int a, score;
  // Data Structure: Vector to store non-increasing participant scores.
  vector<int> arr;
  for (int i = 0; i < n; i++) {
    cin >> a;
    arr.push_back(a);
  }

  /**
   * Functional Utility: Qualification threshold identification.
   */
  score = arr[k - 1];

  /**
   * Logic: Edge case handling for zero-score start.
   * If the highest score is zero, no one qualifies (given non-increasing scores).
   */
  if (arr[0] == 0) {
    cout << 0 << endl;
    return 0;
  }

  /**
   * Block Logic: Evaluates participants from the k-th position onwards to find 
   * the boundary of qualification.
   * Invariant: All participants up to index k-1 are potential qualifiers.
   */
  for (int j = k; j < n; j++) {
    /**
     * Logic: Absolute boundary condition.
     * If the target rank is the last participant, all satisfy the rank constraint.
     */
    if (n == k) {
      cout << k << endl;
      return 0;
    }

    /**
     * Functional Utility: Identifies the first participant who fails the tied-score 
     * or non-zero score constraint.
     * Result: Index 'j' represents the total count of qualifiers.
     */
    if (score != arr[j] && score != 0) {
      cout << j << endl;
      return 0;
    }
  }
  
  // Fallback: Default result if no early termination occurs.
  cout << 0 << endl;
  
  return 0;
}
/**
 * Note: The provided source logic contains some redundancies and might not perfectly 
 * align with the standard 158A problem constraints but was documented as implemented.
 */
