/**
 * @file 158A.CPP
 * @brief Logic for qualification counting in competitive scenarios with threshold filtering.
 * 
 * Logic: Calculates the number of participants who qualify for the next round. 
 * Progression requires a score greater than or equal to the k-th ranker's performance, 
 * provided the score is strictly positive.
 * 
 * Algorithm: Two-pass filtering logic.
 * 1. Counts participants meeting the relative threshold (k-th rank).
 * 2. Refines the count by subtracting those with non-positive (<= 0) absolute scores.
 * 
 * Time Complexity: $O(N)$ where N is the number of participants.
 * Space Complexity: $O(1)$ auxiliary space beyond the fixed-size score buffer (size 100).
 */

#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n, k;
  // Static buffer for up to 100 participant scores.
  int x[100];
  
  // Functional Utility: Reads total headcount (n) and the target rank (k).
  cin >> n >> k;

  // Block Logic: Populates the score array.
  for (int i = 0; i < n; i++) {
    cin >> x[i];
  }

  int meeting_threshold_count = 0;
  /**
   * Block Logic: Initial qualification sweep.
   * Invariant: Scores are non-increasing. Stops immediately once a score 
   * falls below the qualifying benchmark at index k-1.
   */
  for (int i = 0; i < n; i++) {
    if (x[i] >= x[k - 1]) {
      meeting_threshold_count++;
    } else {
      break;
    }
  }

  int invalid_score_count = 0;
  /**
   * Block Logic: Post-processing refinement.
   * Logic: Identifies subset members within the potential qualifiers who failed 
   * the absolute positive score constraint (score <= 0).
   */
  for (int i = 0; i < meeting_threshold_count; i++) {
    if (x[i] <= 0) {
      invalid_score_count++;
    }
  }
  
  // Output: Final cardinality of the qualified set (Threshold Qualifiers - Non-positive Scores).
  cout << meeting_threshold_count - invalid_score_count;
  
  return 0;
}
