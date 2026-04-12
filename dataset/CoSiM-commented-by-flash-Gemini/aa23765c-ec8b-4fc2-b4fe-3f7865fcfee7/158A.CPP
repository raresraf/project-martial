/**
 * @file 158A.CPP
 * @brief Optimized logic for qualification counting in competitive scenarios.
 * 
 * Logic: Calculates the number of participants who qualify for the next round. 
 * Qualification is based on the k-th finisher's score, but requires a strictly 
 * positive result and includes all participants tied with the k-th finisher.
 * 
 * Algorithm: Pivot-based index adjustment.
 * Time Complexity: $O(N)$ where N is the participant count.
 * Space Complexity: $O(1)$ auxiliary space beyond the fixed-size input buffer.
 */

#include <iostream>

using namespace std;

int main() {
  int n, k;
  // Static buffer for up to 110 participant scores.
  int a[110];
  
  // Functional Utility: Reads total participants (n) and the reference rank (k).
  cin >> n >> k;
  
  // Block Logic: Populates the score array.
  for (int i = 0; i < n; i++)
    cin >> a[i];

  /**
   * Functional Utility: Qualification threshold adjustment.
   */
  if (a[k - 1] > 0) {
    /**
     * Case 1: The k-th score is positive.
     * Logic: Increments the qualifying count 'k' to include all participants 
     * who have the same score as the k-th participant.
     */
    while (k < n && a[k] == a[k - 1])
      k++;
  } else {
    /**
     * Case 2: The k-th score is non-positive.
     * Logic: Decrements the qualifying count 'k' until a positive score is reached, 
     * as zero or negative scores do not qualify.
     */
    while (k > 0 && a[k - 1] <= 0)
      k--;
  }
  
  // Output the final count of qualified individuals.
  cout << k << endl;
  
  return 0;
}
