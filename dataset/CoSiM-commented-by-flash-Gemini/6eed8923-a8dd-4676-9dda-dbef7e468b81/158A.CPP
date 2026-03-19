/**
 * @file 158A.CPP
 * @brief Solution to Codeforces Problem 158A: "Next Round".
 * Counts the number of participants who will advance to the next round of a competition.
 * Participants advance if their score is positive and greater than or equal to the score
 * of the participant in the k-th place.
 * Algorithm: Reads all scores, identifies the k-th score as the threshold, then iterates
 * through all scores to count those meeting the criteria. This approach is efficient
 * as it directly uses the property that input scores are provided in non-increasing order,
 * avoiding unnecessary sorting.
 * Time Complexity: O(N), where N is the number of participants, due to two linear passes
 *                  for input and counting.
 * Space Complexity: O(N) to store participant scores.
 */
#include <stdio.h> // Required for standard input/output functions like scanf and printf.

/**
 * @brief Main function to solve the "Next Round" problem.
 * Reads participant scores, determines the k-th place score, and counts qualifiers.
 */
int main() {
  int a[100]; // Array to store the scores of up to 100 participants.
  int n;      // Total number of participants.
  int j, p = 0; // 'j': represents the k-th place (from problem statement 'k'),
                // 'p': counter for the number of participants who advance.

  // Reads the total number of participants (n) and the k-th place (j, equivalent to k).
  scanf("%d%d", &n, &j);

  /**
   * Block Logic: Reads 'n' scores from standard input and stores them in array 'a'.
   * Pre-condition: 'n' specifies the exact number of scores to read.
   * Invariant: After each iteration, 'i+1' scores have been successfully read into 'a'.
   */
  for (int i = 0; i < n; i++) {
    scanf("%d", &a[i]);
  }

  /**
   * Block Logic: Iterates through all participant scores to determine who advances.
   * Qualification Criteria: A participant advances if their score is positive AND
   *                         is greater than or equal to the score of the participant
   *                         at the j-th (k-th) place.
   * Invariant: 'p' accurately counts the number of qualifying participants.
   */
  for (int i = 0; i < n; i++) {
    // Checks if the current participant's score meets both conditions:
    // 1. Score is greater than or equal to the k-th place score (a[j-1]).
    //    Assumes scores are given in non-increasing order, making a[j-1] the cutoff.
    // 2. Score is strictly positive (a[i] > 0).
    if (a[i] >= a[j - 1] && a[i] > 0) {
      p++; // Increment the counter for advancing participants.
    }
  }

  // Prints the total number of participants who advance, followed by a newline.
  printf("%d\n", p);

  return 0;
}
