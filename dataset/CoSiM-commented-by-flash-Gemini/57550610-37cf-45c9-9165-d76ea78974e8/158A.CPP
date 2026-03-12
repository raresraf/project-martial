/**
 * @57550610-37cf-45c9-9165-d76ea78974e8/158A.CPP
 * @brief This program determines the number of participants who advance to the next round of a competition.
 * Participants qualify if their score is greater than or equal to the score of the participant
 * at the k-th place, provided their score is strictly positive. Scores are assumed to be
 * provided in non-increasing order.
 *
 * Algorithm: Identifies the cutoff point for positive scores and then counts participants
 * meeting the score threshold relative to the k-th ranked participant.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(N) for storing participant scores.
 */

#include <iostream> // For standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace to avoid prefixing std::.

int main() {
  int n, k; // Declares 'n' for total participants and 'k' for the target rank.
  cin >> n >> k; // Reads 'n' and 'k' from standard input.

  int scores[n]; // Declares a dynamic array to store the scores of 'n' participants.
  // Initializes 'flag' to a value greater than 'n'. This 'flag' will store the index
  // of the first non-positive score or 'n' if all scores are positive.
  int flag = n + 1; 

  /**
   * Block Logic: Reads 'n' participant scores into the 'scores' array.
   * Invariant: After this loop, 'scores' array contains all participant scores.
   */
  for (int i = 0; i < n; i++) {
    cin >> scores[i];
  }

  /**
   * Block Logic: Finds the index of the first participant with a non-positive score.
   * Invariant: 'flag' will hold the 0-indexed position of the first score <= 0,
   * or it will remain 'n+1' if all scores are positive.
   */
  for (int i = 0; i < n; i++) {
    // Inline: If a score is non-positive, its index is stored in 'flag', and the loop breaks.
    if (scores[i] <= 0) {
      flag = i;
      break;
    }
  }

  /**
   * Block Logic: Determines the number of qualifiers based on the k-th participant's score
   * and the 'flag' indicating the first non-positive score.
   * Pre-condition: 'scores' array is populated and 'flag' is set.
   * Invariant: The correct number of qualifiers is printed.
   */
  if (k < flag) { // Case 1: The k-th participant has a positive score (k is before the first non-positive score).
    int i = 0;    // 'i' will count additional participants with the same score as the k-th participant.
    // Block Logic: Counts participants from the k-th position onwards who have the exact same score as the k-th participant.
    // The loop continues as long as 'i' doesn't cause an out-of-bounds access and the scores match.
    do {
      // Inline: Compares the score of the current participant (relative to k) with the k-th participant's score.
      if (scores[k + i - 1] == scores[k - 1])
        i++; // Increment 'i' if scores match.
      else
        break; // Break if scores do not match (assuming non-increasing order, subsequent scores will also not match).
    } while (i <= (n - k)); // Loop condition: ensure 'k + i - 1' does not exceed 'n-1'.
    
    // The number of qualifiers is k (initial top k participants) + i (additional participants with same score) - 1 (adjustment for loop logic).
    // Or, more simply, it's the index of the first person who doesn't qualify (k+i-1) + 1.
    cout << k + i - 1; 
  } else { // Case 2: The k-th participant has a non-positive score, or 'k' is beyond the range of positive scores.
    // In this scenario, only participants with strictly positive scores (up to 'flag' index) qualify.
    cout << flag; // Prints 'flag', which represents the count of strictly positive scores.
  }
  return 0; // Indicates successful program termination.
}
