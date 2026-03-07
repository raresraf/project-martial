/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem that identifies participants
 * who advance to the next round of a competition.
 * Participants advance if their score is greater than or equal to the k-th place finisher's score
 * and their score is strictly positive.
 * Algorithm: Array traversal and conditional counting.
 * Time Complexity: O(N), where N is the number of participants.
 * Space Complexity: O(N) for storing participant scores.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Declares 'n' for the total number of participants and 'k' for the k-th place threshold.
  int n, k;
  // Reads the total number of participants and the k-th place value.
  cin >> n >> k;
  // Declares an array 'a' of size 'n' to store the scores of all participants.
  int a[n];
  
  /**
   * Block Logic: Populates the array 'a' with participant scores.
   * Invariant: At the start of each iteration, 'i' indicates the current index for score input.
   * Post-condition: The array 'a' contains 'n' participant scores.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  // Initializes 'sum' to count the number of participants who qualify for the next round.
  int sum = 0;
  
  /**
   * Block Logic: Iterates through the scores to determine qualifying participants.
   * Invariant: At the start of each iteration, 'i' points to the current participant's score being evaluated.
   * Pre-condition: 'a[k-1]' must contain a valid score for comparison.
   * Post-condition: 'sum' contains the total count of qualified participants.
   */
  for (int i = 0; i < n; i++) {
    // Checks if the current participant's score is greater than or equal to the k-th place score
    // and if the score is positive.
    if (a[i] >= a[k - 1] && a[i] > 0) {
      sum++;
    }
  }
  // Prints the total count of qualified participants.
  cout << sum;
  
  return 0; // Standard practice to indicate successful execution.
}
