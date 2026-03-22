/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces (Problem 158A).
 * @author Sourabh Bollapragada
 * @date 6/3/20
 *
 * @details This program counts the number of participants who advance to the
 * next round of a contest. The rule for advancing is that a participant's score
 * must be positive and greater than or equal to the score of the k-th place
 * participant. The input scores are guaranteed to be in non-increasing order.
 *
 * @note The code uses a Variable Length Array (VLA) `s[n]`, which is a
 * non-standard C++ extension common in competitive programming environments
 * (e.g., supported by GCC) but not part of the C++ standard. The counting
 * logic is also unusually structured, though functionally correct for the problem.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read all `n` scores into an array `s`.
 * 3. A counter `c` is used to track advancing participants.
 * 4. The code iterates from the first participant. The logic effectively does the following:
 *    a. All participants from 1 to `k` with positive scores are counted.
 *    b. Any participant after the k-th position is only counted if their score
 *       is positive AND equal to the k-th participant's score.
 *
 * @section complexity
 * - Time Complexity: O(N), for reading the N scores and one pass to count them.
 * - Space Complexity: O(N), due to storing all N scores in an array.
 */
#include <iostream>

int main() {
  int n, k;
  std::cin >> n >> k;
  
  // A Variable Length Array (VLA), a non-standard C++ extension.
  int s[n];

  /**
   * @brief Reads all n participant scores into the array.
   */
  for (int i = 0; i < n; i++) {
    std::cin >> s[i];
  }

  int c = 0;
  /**
   * @brief Counts the number of participants who advance.
   * This loop implements the "next round" criteria using a complex conditional.
   */
  for (int i = 0; i < n; i++) {
    // This condition handles participants who are ranked k or better.
    // It counts them as long as their score is positive. Since the scores are
    // non-increasing, anyone with a positive score in this range will have a
    // score >= s[k-1] (if s[k-1] is also positive).
    if (c < k && s[i] > 0) {
      c++;
    } 
    // This condition handles participants ranked after k.
    // They only advance if their score is positive and ties with the k-th score.
    else if (s[i] > 0 && s[i] == s[k - 1]) {
      c++;
    }
  }

  std::cout << c << std::endl;
  return 0;
}