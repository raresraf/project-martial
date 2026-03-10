/**
 * @file 158A.CPP
 * @brief This program solves the "Next Round" problem (Codeforces 158A).
 *   It counts the number of participants who qualify for the next round of a competition.
 *   Qualification criteria: a participant must score at least as much as the `k`-th placed
 *   participant and must have a positive score.
 *   Problem description: https://codeforces.com/problemset/problem/158/A
 * * Algorithm: Array traversal and conditional counting.
 *   1. Reads `n` participant scores into a fixed-size array, using 1-based indexing.
 *   2. Determines the cutoff score based on the `k`-th participant's score.
 *   3. Iterates through the scores, applying qualification criteria.
 *   Note: The `else break` in the counting loop implicitly assumes that the input scores
 *         are provided in non-increasing order. If scores are not sorted, this logic
 *         will lead to incorrect results, as it might prematurely stop counting qualifiers.
 * Time Complexity: O(N) where N is the number of participants. Input reading and the counting loop are O(N).
 * Space Complexity: O(MAX) for the `arr` array, which is effectively O(1) since `MAX` is a small constant (52).
 */
#include <algorithm> // Included for std::max, std::min, etc., though `max` is not explicitly used here.
#include <iostream>  // Required for std::cin, std::cout.

#define MAX 52 // Defines the maximum number of participants, typically for array sizing in competitive programming.
using namespace std;

/**
 * @brief Main function to count qualifying participants.
 * Functional Utility: Reads participant count, cutoff rank, and scores, then
 *                     applies qualification rules to determine and print the number of qualifiers.
 */
int main() {
  int n; // `n`: total participants.
  int k; // `k`: 1-indexed rank of the participant whose score determines the cutoff.
  int arr[MAX]; // `arr`: Array to store participant scores, using 1-based indexing.
  // Block Logic: Reads `n` and `k`.
  cin >> n >> k;
  /**
   * Block Logic: Reads `n` participant scores from input and stores them in the `arr` array.
   * Pre-condition: `n` is a positive integer, `arr` has enough capacity.
   * Invariant: `arr` is populated with scores (1-indexed).
   */
  for (int i = 1; i <= n; i++) {
    cin >> arr[i];
  }
  // `value`: Stores the cutoff score, which is the score of the `k`-th participant.
  int value = arr[k];
  int count = 0; // Accumulates the count of qualifying participants.
  /**
   * Block Logic: Iterates through all participant scores to check qualification.
   * Pre-condition: `arr` contains all scores, `value` is the cutoff score.
   * Invariant: `count` increments for each participant meeting the criteria.
   * Note on `else break`: This logic implicitly assumes scores are sorted in non-increasing order.
   * If a score does not meet the criteria, the loop terminates, potentially missing other qualifiers.
   * This indicates a flaw if the input scores are not guaranteed to be sorted.
   */
  for (int j = 1; j <= n; j++) {
    /**
     * Conditional Logic: Checks two conditions for qualification:
     *   1. `arr[j] >= value`: The participant's score must be greater than or equal to the cutoff score.
     *   2. `arr[j] > 0`: The participant's score must be positive.
     */
    if (arr[j] >= value && arr[j] > 0)
      count++; // Increments the counter if both conditions are met.
    else
      break; // Breaks loop if a participant doesn't qualify (flawed assumption of sorted input).
  }

  // Block Logic: Prints the total count of qualifying participants.
  cout << count << endl;
  return 0;
}