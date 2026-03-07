/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" competitive programming problem (Codeforces 158A).
 * This program determines the number of participants who will advance to the next round.
 * A participant advances if their score is greater than zero and greater than or equal
 * to the score of the k-th (n-th, in 0-indexed terms) place finisher.
 * Algorithm: Array traversal and conditional counting.
 * Time Complexity: O(K), where K is the number of participants.
 * Space Complexity: O(K) for storing participant scores.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

int main() {
  // Declares 'k' for the total number of participants, 'n' for the 0-indexed rank threshold,
  // 's' to store the score of the n-th participant, 'in' an array to store scores,
  // 'i' as a loop counter, and 'player' to count qualifying participants.
  int k, n, s, in[100], i, player = 0;
  // Reads the total number of participants ('k') and the rank threshold ('n').
  cin >> k >> n;
  
  /**
   * Block Logic: Populates the array 'in' with participant scores.
   * Invariant: At the start of each iteration, 'i' indicates the current index for score input.
   * Pre-condition: 'k' is the total number of scores to read.
   * Post-condition: The array 'in' contains 'k' participant scores.
   */
  for (i = 0; i < k; i++) {
    cin >> in[i];
  }
  
  // Retrieves the score of the n-th participant (0-indexed, so at index n-1).
  s = in[n - 1];
  
  /**
   * Block Logic: Iterates through all participant scores to count those who qualify.
   * Invariant: At the start of each iteration, 'i' points to the current participant's score being evaluated.
   * Pre-condition: 'in' contains valid participant scores, and 's' holds the score of the threshold participant.
   * Post-condition: 'player' holds the total count of qualified participants.
   */
  for (i = 0; i < k; i++) {
    /**
     * Conditional Logic: Checks if the current participant's score qualifies.
     * Qualification criteria: Score is greater than or equal to the threshold score 's'
     * AND the score is strictly positive.
     */
    if (in[i] >= s && in[i] != 0) {
      player++; // Increments the counter for qualifying participants.
    }
  }
  // Prints the final count of qualified participants.
  cout << player;
  
  return 0; // Indicates successful program execution.
}
