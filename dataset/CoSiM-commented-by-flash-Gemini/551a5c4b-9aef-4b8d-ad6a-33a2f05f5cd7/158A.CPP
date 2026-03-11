/**
 * @file 158A.CPP
 * @brief Solution to Codeforces Problem 158A: "Next Round".
 *        The problem involves determining the number of participants who will advance
 *        to the next round of a competition. A participant advances if their score
 *        is greater than or equal to the score of the k-th placed participant
 *        and they have a positive score.
 *
 * Algorithm: Array processing and conditional counting. The program reads all
 *            participant scores, identifies the k-th score as the threshold,
 *            and then iterates through all scores again to count those meeting
 *            the advancement criteria.
 * Time Complexity: O(N) where N is the number of participants, as the solution
 *                  involves two linear passes through the array of scores.
 * Space Complexity: O(N) to store the scores of all participants.
 */
#include <iostream>
using namespace std;
int main() {
  /**
   * Functional Utility: Declaration and initialization of variables.
   * `n`: Total number of participants.
   * `k`: The k-th place participant whose score sets the advancement threshold.
   * `i`: Loop counter.
   * `a`: Counter for participants advancing to the next round.
   */
  int n, k, i, a;
  // Input: Read the total number of participants `n` and the k-th position `k`.
  cin >> n >> k;
  // Functional Utility: Declare a variable-length array (VLA) to store participant scores.
  // Note: VLAs are a C99 feature and not standard C++ but are often supported as an extension.
  int arr[n];
  a = 0; // Initialize the counter for advancing participants.

  /**
   * Block Logic: First pass to read participant scores and handle an edge case.
   * Pre-condition: `n` is the total number of participants.
   * Invariant: `arr` contains scores read from input. `i` is the current participant index.
   */
  for (i = 0; i < n; i++) {
    cin >> arr[i]; // Input: Read the score of the current participant.
    /**
     * Conditional Logic: Handles a specific edge case where the very first participant
     * has a score of 0. In competitive programming, a score of 0 typically means
     * no problems were solved, and such a participant cannot advance. If this happens
     * for the first participant, it implies nobody can advance, so output 0 and exit.
     * This condition `i == 0` is crucial to prevent incorrect early exit for other participants.
     */
    if (arr[i] == 0 && i == 0) {
      cout << 0;
      return 0; // Exit program immediately.
    }
  }

  /**
   * Block Logic: Second pass to count participants who qualify for the next round.
   * Pre-condition: `arr` contains all participant scores, `k` is valid.
   * Invariant: `a` counts participants whose score is >= `arr[k-1]` and > 0.
   */
  for (i = 0; i < n; i++) {
    /**
     * Conditional Logic: If a participant's score is less than the k-th score,
     * it means subsequent participants (if scores are non-increasing) also won't qualify.
     * The loop can be optimized by breaking early if scores are sorted, but here
     * it checks individual scores against the threshold.
     */
    if (arr[i] < arr[k - 1])
      break; // Stop counting if a score is below the threshold.
    /**
     * Conditional Logic: If the participant's score meets the threshold AND is positive.
     * Participants with a score of 0 cannot advance, even if `arr[k-1]` is also 0.
     */
    else if (arr[i] >= arr[k - 1] && arr[i] != 0)
      a++; // Increment count of advancing participants.
  }
  // Output: Print the total number of participants advancing to the next round.
  cout << a;
  return 0;
}
