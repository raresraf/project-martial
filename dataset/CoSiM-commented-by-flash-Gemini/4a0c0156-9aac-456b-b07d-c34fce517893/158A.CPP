/**
 * @4a0c0156-9aac-456b-b07d-c34fce517893/158A.CPP
 * @brief This program counts participants who advance in a competition based on
 *        their score relative to the K-th place finisher, requiring a positive score.
 * Algorithm: Single pass (linear scan) through participant scores, dynamically
 *            determining the cutoff score and counting qualifiers.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(1) for storing a few integer variables.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Declare variables:
  // n: Total number of participants.
  // k: The k-th place to consider for the cutoff score.
  int n, k;
  // Functional Utility: Read the total number of participants (n) and the k-th place (k).
  cin >> n >> k;
  // input: Temporary variable to store each participant's score as it's read.
  // advance: Counter for the number of participants who qualify.
  // temp: Stores the cutoff score (score of the K-th place finisher).
  int input, advance = 0, temp;
  // Block Logic: Iterate through all participants' scores.
  for (int i = 0; i < n; i++) {
    // Functional Utility: Read the score of the current participant.
    cin >> input;
    // Pre-condition: If the current participant is before the K-th place AND has a positive score.
    if (i < k - 1 && input > 0) {
      advance++; // Action: Increment the qualifying count.
    }
    // Pre-condition: If the current participant is exactly at the K-th place AND has a positive score.
    else if (i + 1 == k && input > 0) {
      temp = input; // Action: Set this participant's score as the cutoff score.
      advance++;    // Action: Increment the qualifying count.
    }
    // Pre-condition: If the current participant's score is greater than or equal to the cutoff score ('temp').
    // This condition will only be relevant for participants after the K-th place if temp was set.
    // If temp was not set (e.g., K-th place had 0 score), this logic might be problematic.
    else if (input >= temp) {
      advance++; // Action: Increment the qualifying count.
    }
  }
  // Functional Utility: Output the total number of qualifying participants.
  cout << advance;

  return 0; // Indicate successful program termination.
}
