/**
 * @file 158A.CPP
 * @brief Logic for determining competition qualification based on percentile-like benchmarks.
 * 
 * Logic: Calculates the count of participants who progress to the next round. 
 * Progression depends on two criteria:
 * 1. Score must be greater than or equal to the score of the k-th finisher.
 * 2. Score must be strictly positive.
 * 
 * Algorithm: Linear sweep with threshold filtering.
 * Time Complexity: $O(N)$ where N is the participant count.
 * Space Complexity: $O(N)$ for the static array of scores.
 */

#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
  int n, k, c = 0;
  // Functional Utility: Reads total participant count (n).
  cin >> n;
  
  // Data Structure: Static buffer for N scores.
  int a[n];
  // Functional Utility: Reads the qualification rank (k).
  cin >> k;
  
  // Block Logic: Populates the score array.
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  /**
   * Functional Utility: Extracts the reference score benchmark (k-th rank).
   * Note: Array uses 0-based indexing, so rank 'k' is at index 'k-1'.
   */
  int val = a[k - 1];

  /**
   * Block Logic: Evaluates participants against the qualification constraints.
   * Condition 1: Score must meet or exceed the k-th finisher's performance (val).
   * Condition 2: Score must be greater than zero.
   */
  for (int i = 0; i < n; i++) {
    if (a[i] >= val && a[i] > 0) {
      c++;
    }
  }
  
  // Output the final tally of qualified participants.
  cout << c << endl;
  
  return 0;
}
