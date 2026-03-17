/**
 * @69b671dc-bfc9-4a84-8bab-e66a8cff6a5a/158A.CPP
 * @brief Problem: Next Round.
 *        Determines the number of participants who advance to the next round of a competition.
 *        Participants advance if their score is greater than or equal to the k-th place finisher's score
 *        and their score is strictly positive.
 * Algorithm: Array traversal and conditional counting.
 *            1. Read the total number of participants ('x') and the k-th position ('y').
 *            2. Read all participant scores into an array.
 *            3. Iterate through the scores. For each score, check if it meets the advancement criteria:
 *               a. Score is positive.
 *               b. Score is greater than or equal to the score of the k-th place participant.
 *            4. Count the number of participants satisfying both conditions.
 * Time Complexity: O(x) where 'x' is the number of participants.
 *                  The code iterates through the array of 'x' participants twice (once for input, once for counting).
 * Space Complexity: O(x) for storing the 'x' participant scores in an array.
 */
#include <stdio.h>  // Functional Utility: Provides standard input/output functions like scanf and printf.
#include <string.h> // Functional Utility: Not used in this specific problem, but commonly included for string operations.

int main() {
  int x, y, i, cou = 0; // Invariant: 'cou' stores the count of advancing participants.
  // Block Logic: Reads the total number of participants (x) and the k-th position (y).
  scanf("%d %d", &x, &y);
  int c[x]; // Functional Utility: Declares a dynamic array to store participant scores.
  
  // Block Logic: Reads participant scores into the array 'c'.
  // Pre-condition: 'x' has been read and represents the number of participants.
  // Invariant: After this loop, 'c[i]' contains the score of participant 'i'.
  for (i = 1; i <= x; i++) // Note: Array indexing starts from 1, typical in some competitive programming contexts.
    scanf("%d", &c[i]);
  
  // Block Logic: Counts participants who advance to the next round.
  // Pre-condition: 'c' array is populated with scores, and 'y' (k-th position) is valid.
  // Invariant: 'cou' is incremented for each participant whose score is positive AND
  //            greater than or equal to the k-th place participant's score.
  for (i = 1; i <= x; i++) {
    // Block Logic: Checks the advancement criteria for the current participant.
    // Pre-condition: 'c[i]' is the current participant's score. 'c[y]' is the k-th place score.
    // Post-condition: 'cou' is incremented if the criteria are met.
    if (c[i] > 0 && c[i] >= c[y])
      cou = cou + 1; // Functional Utility: Increments the counter for advancing participants.
  }
  // Functional Utility: Prints the final count of participants who advance.
  printf("%d", cou);

  return 0; // Functional Utility: Indicates successful program execution.
}
