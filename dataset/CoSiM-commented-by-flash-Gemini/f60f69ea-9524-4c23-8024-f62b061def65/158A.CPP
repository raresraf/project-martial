/**
 * @file 158A.CPP
 * @brief This program determines the number of participants who advance to the next round of a competition.
 * Participants advance if their score is greater than or equal to the score of the k-th place finisher
 * and their score is positive.
 *
 * Algorithm: Array traversal and conditional counting.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(N) to store the participants' scores.
 */

#include <bits/stdc++.h> // Includes iostream, vector, etc.

using namespace std;

int main() {
  // Functional Utility: 'c' counts the number of participants who advance.
  int c = 0;

  // Functional Utility: 'k' represents the k-th place finisher.
  int k;
  // Functional Utility: 'n' represents the total number of participants.
  int n;
  // Functional Utility: 'a' is an array to store the scores of up to 51 participants.
  int a[51];
  // Functional Utility: 'i' is a loop counter.
  int i;

  // Block Logic: Reads the total number of participants 'n' and the k-th place 'k' from standard input.
  cin >> n >> k;

  /**
   * Block Logic: Reads the scores of all 'n' participants into the array 'a'.
   * Invariant: After each iteration, 'a[i]' holds the score of the i-th participant.
   */
  for (i = 0; i < n; i++) {
    cin >> a[i];
  }

  /**
   * Block Logic: Iterates through the participants' scores to determine who advances.
   * Invariant: 'i' represents the current participant being evaluated. The loop breaks
   *            if a participant does not meet the criteria, assuming scores are sorted or
   *            that this problem implies a specific order of input.
   */
  for (i = 0; i < n; i++) {
    /**
     * Block Logic: Checks two conditions for advancement:
     * 1. The participant's score `a[i]` must be positive.
     * 2. The participant's score `a[i]` must be greater than or equal to the k-th place finisher's score `a[k-1]`.
     * Pre-condition: 'a' contains all scores, and 'k' is a valid index for `a[k-1]`.
     * Invariant: If both conditions are met, the counter 'c' is incremented.
     */
    if (a[i] > 0 && a[i] >= a[k - 1]) {
      c++;
    } else {
      // Inline: If a participant does not meet the criteria, no further participants
      //         will meet it (assuming a non-increasing order of relevant scores).
      break;
    }
  }
  // Output the total count of participants who advance to the next round.
  cout << c;
}
