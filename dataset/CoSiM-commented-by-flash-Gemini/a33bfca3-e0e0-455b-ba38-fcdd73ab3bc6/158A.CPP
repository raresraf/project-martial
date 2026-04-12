/**
 * @file 158A.CPP
 * @brief Logic for qualifying participants for the next round based on score thresholds.
 * 
 * Algorithm: Percentile-based filtering using the k-th highest value.
 * Logic: Sorts scores to identify the threshold at rank K and counts all participants 
 * who meet this threshold and have a strictly positive score.
 * 
 * Time Complexity: $O(N \log N)$ due to sorting.
 * Space Complexity: $O(N)$ for the dynamic array allocation.
 */

#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n, k;
  // Functional Utility: Reads the total participant count and the target qualification rank.
  cin >> n >> k;
  
  // Dynamic Memory Allocation: Stores N scores in a heap-allocated buffer.
  int *scores = new int[n];
  for (int u = 0; u < n; ++u) {
    cin >> scores[u];
  }

  // Logic: Orders scores ascending to simplify rank-based threshold identification.
  sort(scores, scores + n);

  /**
   * Functional Utility: Identifies the qualification threshold (k-th highest score).
   * Note: In an ascending array of size N, the k-th highest is at index (n - k).
   */
  int min_qualifying_score = scores[n - k];
  int promoted = 0;

  /**
   * Block Logic: Evaluates participants from highest to lowest score.
   * Invariant: Each promoted participant must satisfy (Score >= threshold) AND (Score > 0).
   * Constraint: Terminate if the participant does not meet the criteria or all are processed.
   */
  while (promoted < n && scores[n - 1 - promoted] > 0 &&
         scores[n - 1 - promoted] >= min_qualifying_score) {
    ++promoted;
  }
  
  // Output the final count of qualified individuals.
  cout << promoted << endl;
  
  return 0;
}
