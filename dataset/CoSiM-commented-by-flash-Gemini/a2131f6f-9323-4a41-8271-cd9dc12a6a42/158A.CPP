/**
 * @file 158A.CPP
 * @brief Logic for qualifying participants for the next round based on score thresholds.
 * 
 * Logic: Calculates the number of participants whose scores are greater than or equal 
 * to the k-th ranker's score, provided their score is strictly positive.
 * 
 * Algorithm: Single-pass threshold filtering.
 * Time Complexity: $O(N)$ where N is the number of participants.
 * Space Complexity: $O(N)$ for score storage in a contiguous buffer.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k;
  // Functional Utility: Reads the total participant count and the target qualification rank.
  cin >> n >> k;
  
  // Array allocated for N scores using 1-based indexing for logic alignment.
  int arr[n + 1];
  for (int i = 1; i <= n; i++) {
    cin >> arr[i];
  }

  /**
   * Functional Utility: Extracts the reference qualification score from the k-th rank.
   */
  int reqScore = arr[k], count = 0;

  /**
   * Block Logic: Evaluates each score against the dynamic qualification threshold.
   * Condition 1: Score must be greater than zero (strict positive).
   * Condition 2: Score must meet or exceed the k-th participant's result.
   */
  for (int i = 1; i <= n; i++) {
    if ((arr[i] > 0) && (arr[i] >= reqScore)) {
      count += 1;
    }
  }
  
  // Output the total count of qualified individuals.
  cout << count << "\n";
  
  return 0;
}
