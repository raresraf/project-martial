//
// Created by Sourabh Bollapragada on 6/3/20.
//

// Problem Link https://codeforces.com/problemset/problem/158/A

/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" competitive programming problem (Codeforces 158A).
 * This program determines the number of participants who will advance to the next round.
 * A participant advances if their score is greater than zero and greater than or equal
 * to the score of the k-th place finisher.
 * Algorithm: Array traversal and conditional counting with specific advancement rules.
 * Time Complexity: O(N), where N is the number of participants.
 * Space Complexity: O(N) for storing participant scores.
 */
#include <iostream>

int main() {
  // Declares 'n' for the total number of participants and 'k' for the k-th place threshold.
  int n, k;
  // Reads the total number of participants and the k-th place value.
  std::cin >> n >> k;
  // Declares an array 's' of size 'n' to store the scores of all participants.
  int s[n];

  /**
   * Block Logic: Populates the array 's' with participant scores.
   * Invariant: At the start of each iteration, 'i' indicates the current index for score input.
   * Post-condition: The array 's' contains 'n' participant scores.
   */
  for (int i = 0; i < n; i++) {
    std::cin >> s[i];
  }

  // Initializes 'c' to count the number of participants who qualify for the next round.
  int c = 0;
  /**
   * Block Logic: Iterates through the scores to determine qualifying participants based on problem rules.
   * Invariant: At the start of each iteration, 'i' points to the current participant's score being evaluated.
   * Pre-condition: 's' contains valid participant scores, and 'k' is a valid rank (1 <= k <= n).
   * Post-condition: 'c' holds the total count of qualified participants.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Conditional Logic: Determines if the current participant qualifies.
     * The logic here counts participants who are either among the top 'k'
     * and have a positive score, OR are beyond the top 'k' but still
     * match the score of the k-th participant and have a positive score.
     */
    if (c < k && s[i] > 0) {
      // If the current count 'c' is less than 'k' (meaning we are still filling top 'k' spots)
      // AND the current participant has a positive score, they qualify.
      c++;
    } else if (s[i] > 0 && s[i] == s[k - 1]) {
      // If the participant has a positive score AND their score is equal to the
      // score of the k-th place participant (s[k-1]), they also qualify.
      c++;
    }
  }

  // Prints the final count of qualified participants.
  std::cout << c << std::endl;
  
  return 0; // Standard practice to indicate successful execution.
}