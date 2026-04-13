/**
 * @file 158A.CPP
 * @brief This program solves the "Next Round" problem. It counts how many participants
 * will advance to the next round of a competition. Participants advance if their score
 * is greater than or equal to the score of the participant who finished in the k-th place,
 * and their score is strictly positive. This solution handles different cases based on the
 * k-th place score.
 * Algorithm: Read scores, then branch logic based on whether the k-th score is zero or positive,
 * and count qualifying participants.
 * Time Complexity: O(N), where N is the number of participants, due to single pass for input
 * and additional passes in the conditional logic (at most N).
 * Space Complexity: O(1) for a fixed-size array (max 50 participants assumed).
 */
#include <stdio.h> // Required for standard input/output functions like scanf and printf.

int main(void) {

  int n; // Stores the total number of participants.
  int k; // Stores the 1-indexed position for the k-th place.

  int a[50]; // A fixed-size array to store the scores of up to 50 participants.

  // Reads the total number of participants 'n' and the k-th place 'k'.
  scanf("%d %d", &n, &k);
  /**
   * Block Logic: Reads 'n' participant scores from standard input and stores them in the 'a' array.
   * Invariant: After this loop, 'a' contains all participant scores in the order they were provided.
   */
  for (int i = 0; i < n; i++) {
    scanf("%d ", &a[i]);
  }

  /**
   * Conditional Logic: Determines the counting strategy based on the score of the participant
   * at the (k-1)th index (k-th place, 0-indexed).
   */
  if (a[k - 1] == 0) { // If the k-th participant's score is 0.
    // This block handles cases where the k-th score is zero.
    // In this scenario, only participants with a positive score can advance.
    if (a[0] == 0) { // Edge case: If even the first participant scored 0.
      printf("%d", a[0]); // Prints 0 as no one can advance.
      return 0;            // Terminates the program.
    }
    /**
     * Block Logic: Counts participants with positive scores when the k-th score is 0.
     * It iterates backward from the (k-1)th position to find the first positive score.
     * All participants from index 0 up to that point qualify.
     */
    for (int j = k - 1; j >= 0; j--) {
      if (a[j] != 0) { // Finds the rightmost participant with a positive score.
        printf("%d", j + 1); // Prints the count (j+1 for 1-indexed count).
        return 0;             // Terminates the program.
      }
    }
  } else { // If the k-th participant's score is positive.
    // This block handles cases where the k-th score is positive.
    // Participants with scores equal to or greater than this score (and positive) qualify.
    /**
     * Block Logic: Iterates from the k-th participant's position (0-indexed) to the end of the array.
     * It counts all participants whose score is greater than the next participant's score, effectively
     * finding the boundary of those with scores greater than or equal to the k-th score.
     */
    for (int j = k - 1; j < n; j++) {
      // If the current participant's score is greater than the next participant's score,
      // it means we've passed the group of participants with scores >= k-th score.
      // This logic is intended to find the first participant whose score drops below the k-th score,
      // and print the count up to that point.
      if (a[j] > a[j + 1]) {
        printf("%d", j + 1); // Prints the count (j+1 for 1-indexed count).
        return 0;             // Terminates the program.
      }
    }
    // If the loop completes, it means all participants from k-th place onwards
    // have scores equal to the k-th score, or no one has a lower score.
    // In this scenario, all 'n' participants potentially qualify.
    printf("%d", n); // Prints the total number of participants 'n'.
  }
  // Standard program exit (unreachable in current logic due to return 0 inside if/else blocks).
  return 0;
}
