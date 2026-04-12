/**
 * @file 158A.CPP
 * @brief Logic for determining qualification benchmarks in a multi-participant competition.
 * 
 * Logic: Calculates the cardinality of the subset of participants who meet or exceed 
 * the k-th rank's score, with the additional constraint that the score must be 
 * strictly positive.
 * 
 * Algorithm: Single-pass threshold filtering using a vector-backed sequence.
 * Time Complexity: $O(N)$ where N is the participant count.
 * Space Complexity: $O(N)$ for participant score storage.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k;
  // Functional Utility: Reads global participant count (n) and qualification rank (k).
  cin >> n >> k;
  
  // Vector to store the sequence of participant scores.
  vector<int> a;
  for (int i = 0; i < n; i++) {
    int temp;
    cin >> temp;
    a.push_back(temp);
  }

  /**
   * Functional Utility: Extracts the reference score benchmark from the k-th participant.
   * Note: Vector uses 0-based indexing, so rank 'k' is at index 'k-1'.
   */
  int qualifying_score = a[k - 1];
  int count = 0;

  /**
   * Block Logic: Evaluates each individual score against the qualification criteria.
   * Condition 1: Score (x) must be >= the k-th ranker's score (qualifying_score).
   * Condition 2: Score (x) must be greater than zero.
   */
  for (int x : a) {
    if (x >= qualifying_score && x > 0)
      count++;
  }
  
  // Output the total number of qualified participants.
  cout << count << endl;
  
  return 0;
}
