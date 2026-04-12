/**
 * @file 158A.CPP
 * @brief Logic for determining competition qualification using binary search benchmarks.
 * 
 * Logic: Calculates the number of participants qualifying for the next round. 
 * Advancement is based on meeting or exceeding a reference score (k-th rank), 
 * provided the score is strictly positive.
 * 
 * Algorithm: Sorting followed by `lower_bound` search.
 * Time Complexity: $O(N \log N)$ where N is the participant count.
 * Space Complexity: $O(N)$ for participant score storage.
 */

#include <bits/stdc++.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  // Data Structure: Vector to store participant scores for rank-based analysis.
  vector<int> v; 
  int n, t, k;
  
  // Functional Utility: Reads total headcount (n) and the reference rank (k).
  if (scanf("%d", &n) != 1) return 0;
  cin >> k;

  // Block Logic: Populates the score vector.
  for (int i = 0; i < n; i++) {
    cin >> t;
    v.push_back(t);
  }

  /**
   * Functional Utility: Extracts the reference benchmark score.
   * Note: uses 0-indexed access v[k], which corresponds to rank (k+1) if input is 1-indexed.
   */
  int threshold_score = v[k];

  /**
   * Logic: Absolute qualification constraint.
   * Progression requires a strictly positive score. If the benchmark is 0 or less, 
   * the code terminates early with 0 qualifiers.
   */
  if (threshold_score <= 0) {
    std::cout << "0" << std::endl;
    return 0;
  }

  // Logic: Orders the scores ascending to enable efficient binary search for the threshold.
  sort(v.begin(), v.end());

  /**
   * Functional Utility: Identifies the subset meeting the qualification criteria.
   * Algorithm: `lower_bound` finds the first element >= threshold_score.
   * Result: The distance from the lower bound to the end of the vector is the 
   * count of qualifying participants.
   */
  auto it = lower_bound(v.begin(), v.end(), threshold_score);
  
  // Output the total number of qualified individuals.
  cout << v.end() - it;
  
  return 0;
}
