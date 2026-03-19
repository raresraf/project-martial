/**
 * @file 158A.CPP
 * @brief Solution to Codeforces Problem 158A: "Next Round".
 *
 * Counts the number of participants who will advance to the next round of a competition.
 * A participant advances if their score is positive and is greater than or equal to
 * the score of the participant in the k-th place (k-th highest score).
 *
 * Algorithm: Read scores sequentially. Dynamically identify the k-th highest positive score.
 * Count all subsequent scores that are equal to this k-th score (and positive).
 * This solution leverages the problem constraint that scores are given in non-increasing order.
 *
 * Time Complexity: O(N), where N is the number of participants, due to a single pass
 *                  through the input scores.
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++, like iostream.
using namespace std;

/**
 * @brief Main function to solve the "Next Round" problem.
 * Reads participant scores, determines the k-th place score dynamically, and counts qualifiers.
 */
int main() {
  int n, k;     // 'n': total participants, 'k': the k-th place rank.
  int a;        // Temporary variable to read each participant's score.
  int current;  // Stores the score of the k-th place participant.
  int result = 0; // Counter for the number of participants who advance.
  
  cin >> n >> k; // Reads the total number of participants (n) and the k-th place (k).

  /**
   * Block Logic: Iterates through each participant's score to determine qualifiers.
   * Assumes scores are provided in non-increasing order as per typical problem statements.
   * Pre-condition: 'n' is the number of participants, 'k' is the target rank.
   * Invariant: 'result' accurately counts qualified participants, and 'current' holds
   *            the k-th place score (or similar threshold).
   */
  for (int i = 0; i < n; i++) {
    cin >> a; // Reads the score of the current participant.
    
    /**
     * Block Logic: If the current score is zero, no one else can advance
     *              (scores are non-increasing and must be positive).
     */
    if (a == 0) {
      break; // Terminate early as no more positive scores are expected.
    }
    
    /**
     * Block Logic: Handles participants up to the k-th place.
     * These participants (if their score is positive) automatically qualify,
     * and their score defines the minimum threshold for others.
     */
    if (result < k) {
      result++;     // Increment result, counting this participant.
      current = a;  // Set this score as the k-th place score.
    } else {
      /**
       * Block Logic: Handles participants after the k-th place has been established.
       * They qualify only if their score matches the k-th place score.
       */
      if (current == a) {
        result++; // This participant also qualifies.
      } else {
        // If the current score is less than the k-th place score,
        // no subsequent participants (due to non-increasing order) will qualify.
        break; // Terminate early.
      }
    }
  }
  
  cout << result; // Prints the total number of participants who advance.
  return 0;
}
