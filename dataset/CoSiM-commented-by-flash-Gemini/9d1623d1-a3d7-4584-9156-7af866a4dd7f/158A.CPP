/**
 * @file 158A.CPP
 * @brief This program determines the number of participants who will advance to the next round
 * of a competition. The qualification rule states that a participant advances if their
 * score is greater than or equal to the score of the k-th placed participant,
 * and their score is strictly positive.
 *
 * Algorithm: Array storage and two-pass linear scan.
 * 1. Read the total number of participants (n) and the significant rank (k).
 * 2. Read all 'n' participant scores into an array.
 * 3. Identify the score of the k-th participant.
 * 4. Iterate through all participant scores again, counting how many meet both criteria:
 *    a. Score is greater than or equal to the k-th participant's score.
 *    b. Score is strictly greater than zero.
 * 5. Print the final count.
 *
 * Time Complexity: O(N), where N is the number of participants.
 * This involves one pass to read scores and another pass to count qualified participants.
 *
 * Space Complexity: O(N) to store the 'n' participant scores in an array.
 */

#include <iostream>

using namespace std;

/**
 * @brief Main function orchestrating the participant qualification logic.
 * Reads input, processes scores based on competition rules, and outputs the count
 * of participants advancing to the next round.
 * @pre Standard input expects:
 *      - An integer `n` (1 <= k <= n <= 50): the total number of participants.
 *      - An integer `k` (1 <= k <= n): the rank of the participant whose score
 *        determines the cutoff.
 *      - `n` space-separated integers: the scores of participants (0 <= score <= 100).
 * @post Prints a single integer to standard output, representing the number of
 *       participants who qualified.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: 'n' stores the total number of participants, 'k' stores the rank cutoff,
  // 'c' counts the qualified participants.
  int n, k, c = 0;
  // Functional Utility: Reads 'n' and 'k' from standard input.
  cin >> n >> k;
  // Functional Utility: Declares an array to store 'n' participant scores.
  // Note: Array `participantes` is declared with size `n` but will be accessed
  // using 1-based indexing in the loops, which is a common pattern in
  // competitive programming but can lead to out-of-bounds access if not
  // handled carefully (e.g., if `n` is large and 0-indexing is strictly enforced).
  int participantes[n];
  /**
   * Block Logic: Reads 'n' participant scores and populates the `participantes` array.
   * Invariant: After each iteration `i`, `participantes[i]` contains the score
   *            of the `i`-th participant (using 1-based indexing for conceptual alignment with problem).
   * Pre-condition: Standard input contains 'n' integers following 'n' and 'k'.
   */
  for (int i = 1; i <= n; i++) {
    cin >> participantes[i];
  }
  /**
   * Block Logic: Iterates through all participants to determine qualification based on rules.
   * Invariant: `c` accurately reflects the count of qualified participants among `participantes[1...i-1]`.
   * Pre-condition: `participantes` array is fully populated and `k` is a valid index.
   */
  for (int i = 1; i <= n; i++) {
    /**
     * Block Logic: Checks the primary qualification criterion:
     * Current participant's score must be greater than or equal to the k-th participant's score.
     * Pre-condition: `participantes[i]` and `participantes[k]` hold valid scores.
     */
    if (participantes[i] >= participantes[k]) {
      /**
       * Block Logic: Checks the secondary qualification criterion:
       * The current participant's score must be strictly positive (non-zero).
       * If both criteria are met, increment the qualified counter.
       * Pre-condition: The primary qualification criterion is met.
       */
      if (participantes[i] != 0)
        c++; // Functional Utility: Increments the count of qualified participants.
    }
  }
  // Functional Utility: Prints the final count of qualified participants.
  cout << c;
  return 0;
}

