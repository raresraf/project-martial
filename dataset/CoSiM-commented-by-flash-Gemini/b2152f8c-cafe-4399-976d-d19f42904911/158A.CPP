/**
 * @file 158A.CPP
 * @brief Logic for qualification counting based on rank-derived score thresholds.
 * 
 * Logic: Calculates the number of participants qualifying for the next round. 
 * Progression requires a score >= the k-th finisher's score, and the score 
 * must be strictly positive.
 * 
 * Algorithm: Single-pass threshold filtering using a vector-backed sequence.
 * Time Complexity: $O(N)$ where N is the participant count.
 * Space Complexity: $O(N)$ for participant score storage.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, qualified_count = 0, temp;
  
  // Functional Utility: Reads total headcount (n) and the target qualification rank (k).
  if (!(cin >> n >> k)) return 0;
  
  // Data Structure: Vector to buffer all participant scores.
  vector<int> a;
  for (int i = 0; i < n; ++i) {
    cin >> temp;
    a.push_back(temp);
  }

  /**
   * Functional Utility: Extracts the qualifying benchmark score.
   * Note: Vector uses 0-based indexing, so rank k corresponds to index k-1.
   */
  int qualifying_score = a[k - 1];

  /**
   * Block Logic: Evaluates each score against the qualification constraints.
   * Condition 1: Score must be greater than or equal to the benchmark (qualifying_score).
   * Condition 2: Score must be strictly positive (> 0).
   */
  for (int i = 0; i < n; ++i) {
    if ((a[i] >= qualifying_score) && a[i] > 0) {
      qualified_count++;
    }
  }
  
  // Output the final count of individuals who progress.
  cout << qualified_count;
  
  return 0;
}
