/**
 * @file 158A.CPP
 * @brief Solution attempt for Codeforces problem 158A - Next Round.
 *        The problem typically asks to count participants who advance to the next round.
 *        A participant advances if their score is greater than or equal to the k-th place score,
 *        and their score is positive.
 * Algorithm (Observed):
 *   1. Reads 'n' (total participants) and 'k' (the k-th place index, 1-based).
 *   2. Iterates 'k' times, reading scores. In this loop, 'cur' stores the score of the k-th participant.
 *      `ct` counts how many participants have been processed so far.
 *      If a score of 0 is encountered within these first 'k' scores, it prints `ct` and terminates.
 *   3. After the first 'k' scores, it continues to read subsequent scores until the end of input
 *      or until a score different from 'cur' (the k-th score) is encountered.
 *   4. Prints the final value of `ct`.
 * Deviations/Observations:
 *   - The logic appears to count participants who scored exactly the same as the k-th participant,
 *     rather than those scoring *greater than or equal to* the k-th participant's score.
 *   - The `if (!cur)` check for a zero score within the first 'k' participants immediately terminates
 *     and prints the count so far, which might not be the intended behavior for the problem.
 *   - The problem typically requires checking against the k-th score *after* all scores up to k
 *     have been read and the k-th score is determined, and then comparing all scores against this threshold.
 *   - The standard solution usually involves finding the k-th score and then iterating all scores to compare.
 * Time Complexity: O(N) due to reading N scores.
 * Space Complexity: O(1) as it only stores a few variables.
 */
#include <iostream>
using std::cin;
using std::cout;
int main() {
  /**
   * @brief Main function implementing a solution for the "Next Round" problem.
   *
   * Reads total participants 'n' and a k-th position 'k'. It then processes
   * participant scores to count how many participants advance based on a comparison
   * with the score at the 'k'-th position. The implementation's logic for counting
   * and termination deviates from typical problem interpretations.
   * @return int Returns 0 upon successful execution.
   */
  int n, k, cur = 0, ct = 0, rest;
  cin >> n >> k; // Reads total participants 'n' and the k-th position 'k'.

  // Block Logic: Processes the first 'k' scores to determine the k-th score and count participants.
  // Pre-condition: 'k' is a valid 1-based index within the 'n' participants.
  // Invariant: 'cur' stores the score of the k-th participant, 'ct' counts participants processed.
  //            Early exit if a non-positive score is encountered.
  for (int i = 0; i < k; ++i, ++ct) {
    cin >> cur; // Reads the current participant's score.
    // Block Logic: Checks if the current score is non-positive.
    // If a non-positive score is found within the first 'k' participants,
    // the program terminates and prints the count up to that point.
    if (!cur) {
      cout << ct; // Prints the count of participants processed before the non-positive score.
      return 0;   // Exits the program.
    }
  }
  // Block Logic: Continues counting participants whose scores are equal to the k-th score.
  // Pre-condition: 'cur' holds the k-th participant's score.
  // Invariant: 'ct' is incremented for each subsequent participant whose score matches 'cur'.
  // This loop stops when a different score is read or end of input is reached.
  while (cin >> rest && cur == rest) {
    ++ct;
  }
  cout << ct; // Prints the total count of participants including the k-th and those with same score.
  return 0;
}
