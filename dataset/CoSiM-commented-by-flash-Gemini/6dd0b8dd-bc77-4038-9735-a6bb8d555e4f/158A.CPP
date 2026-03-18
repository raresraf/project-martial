/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem: "Next Round".
 * Counts the number of participants who advance to the next round.
 * Participants advance if their score is positive and greater than or equal to
 * the k-th placed participant's score. This solution implements a specific
 * logic to track the k-th score dynamically.
 * Algorithm: Single pass with conditional logic to track qualifying scores.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(N) to store participant scores.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to solve the Next Round problem.
 * Reads 'n' (total participants) and 'k' (k-th position).
 * Determines the number of participants who achieved a score greater than 0
 * and greater than or equal to the k-th best score.
 */
int main() {
  int n, k;
  cin >> n >> k;
  int arr[n]; // Array to store participant scores.
  /**
   * Block Logic: Reads 'n' participant scores into the array.
   * Pre-condition: 'n' and 'k' have been read.
   * Invariant: After the loop, 'arr' contains all participant scores.
   */
  for (int i = 0; i < n; i++)
    cin >> arr[i];
  int count = 0; // Counter for qualifying participants.

  /**
   * Block Logic: Determines qualifying participants.
   * The first participant's score is used as an initial reference if positive.
   * This structure handles edge cases where the first participant might not qualify.
   * Pre-condition: 'arr' contains scores, 'count' is 0.
   */
  if (arr[0] > 0) {
    int prev = arr[0]; // 'prev' stores the score of the k-th participant (or a higher one if k-th is not yet reached).
    int i = 0;
    int pos = 1; // 'pos' tracks the current logical position.

    /**
     * Block Logic: Iterates through the scores, counting qualifiers.
     * The loop continues as long as there are participants and their score is positive.
     * Invariant: 'count' accumulates qualifying participants; 'prev' updates to the score
     *            of the current k-th qualifying participant or higher if the current one
     *            is lower and within the top 'k'.
     */
    while (i < n) {
      /**
       * Block Logic: If a participant has a non-positive score, no further participants can qualify.
       * Pre-condition: 'arr[i]' is the current participant's score.
       */
      if (arr[i] <= 0)
        break; // Stop if score is not positive.
      /**
       * Block Logic: Updates the reference score if a lower score is encountered within the top 'k'.
       * 'prev' will hold the score of the k-th person *that has a positive score*.
       */
      if (arr[i] < prev) {
        prev = arr[i]; // Update 'prev' to the new, lower score.
        pos = i + 1; // Update 'pos' to the current 1-indexed position.
        /**
         * Block Logic: If the position exceeds 'k' after a score drop, break.
         * This handles cases where a participant outside the top 'k' has a score
         * lower than 'prev' but still qualifies against the original k-th score.
         * However, the logic for 'pos' is more about tracking position, not directly
         * comparing with 'k' in this specific way. This check is crucial for the
         * problem definition: only those *at least* as good as the k-th participant.
         */
        if (pos > k)
          break; // If current position exceeds k, and score dropped, then no more qualifiers.
      }
      count++; // Increment the count of qualifying participants.
      i++;     // Move to the next participant.
    }
  }
  cout << count; // Output the total count of qualifying participants.
  return 0;      // Indicate successful execution.
}