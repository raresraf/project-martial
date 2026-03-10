/**
 * @file 158A.CPP
 * @brief This program solves the "Next Round" problem (Codeforces 158A).
 *   It counts the number of participants who qualify for the next round of a competition.
 *   Qualification criteria: a participant must score at least as much as the `p`-th placed
 *   participant and must have a positive score.
 *   Problem description: https://codeforces.com/problemset/problem/158/A
 * * Algorithm: Array traversal and conditional counting.
 *   1. Reads `n` participant scores into a vector.
 *   2. Determines the cutoff score based on the `p`-th participant's score.
 *   3. Iterates through all scores, applying the qualification criteria to count qualifiers.
 * Time Complexity: O(N) where N is the number of participants, due to two linear passes
 *   (one for reading input, one for counting qualifiers).
 * Space Complexity: O(N) to store the `nums` vector.
 */
#include <iostream> // Required for standard input/output operations.
#include <vector>   // Required for std::vector.

using namespace std;

/**
 * @brief Main function to count qualifying participants.
 * Functional Utility: Reads participant count, cutoff rank, and scores, then
 *                     applies qualification rules to determine and print the number of qualifiers.
 */
int main() {
  int n; // Stores the total number of participants.
  int p; // Stores the 1-indexed rank of the participant whose score determines the cutoff.
  // Block Logic: Reads `n` and `p`.
  cin >> n >> p;
  // Functional Utility: Creates a vector of integers of size `n` to store participant scores.
  vector<int> nums(n);
  /**
   * Block Logic: Reads `n` participant scores from input and stores them in the `nums` vector.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: `nums` vector is populated with all participant scores.
   */
  for (int i = 0; i < n; i++) {
    cin >> nums[i];
  }
  int count = 0; // Accumulates the count of qualifying participants.
  /**
   * Block Logic: Iterates through all participant scores to check qualification.
   * Pre-condition: `nums` contains all scores, and `p` defines the cutoff.
   * Invariant: `count` increments for each participant meeting the criteria.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Conditional Logic: Checks two conditions for qualification:
     *   1. `nums[i] >= nums[p - 1]`: The participant's score must be greater than or equal to
     *      the score of the `p`-th placed participant (using 0-indexed `p - 1`).
     *   2. `nums[i] > 0`: The participant's score must be positive.
     */
    if (nums[i] >= nums[p - 1] && nums[i] > 0)
      count++; // Increments the counter if both conditions are met.
  }
  // Block Logic: Prints the total count of qualifying participants.
  cout << count << endl;
  return 0;
}
