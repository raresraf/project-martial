/**
 * @file 158A.CPP
 * @brief Logic for qualifying participants based on score benchmarks in a competitive environment.
 * 
 * Algorithm: Threshold-based filtering using the k-th ranker's performance.
 * Logic: Identifies the k-th participant's score and counts how many participants 
 * achieved a score >= threshold, provided the score is strictly positive.
 * 
 * Time Complexity: $O(N)$ where N is the number of participants.
 * Space Complexity: $O(N)$ for fixed-size score storage (buffer size 51).
 */

#include <iostream>
using namespace std;

int main() {
  // Static buffer for participant scores.
  int arr[51];
  int n, k, result = 0;
  
  // Functional Utility: Reads the total participant count and the target rank (k).
  cin >> n >> k;

  // Block Logic: Populates the score buffer from the input stream.
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  /**
   * Functional Utility: Normalizes rank 'k' to 0-based indexing and retrieves 
   * the qualifying score benchmark.
   */
  k -= 1;
  int qualifying_benchmark = arr[k];

  /**
   * Block Logic: Evaluates each score against the qualification criteria.
   * Condition 1: Score must be greater than or equal to the k-th rank's score.
   * Condition 2: Score must be non-zero (strictly positive).
   */
  for (int i = 0; i < n; i++) {
    if (arr[i] >= qualifying_benchmark && arr[i] != 0)
      result++;
  }
  
  // Output the final count of qualified participants.
  cout << result;
  
  return 0;
}
