/**
 * @file 158A.CPP
 * @brief Solution for Codeforces Problem 158A - "Next Round".
 * 
 * @details
 * This program determines the number of participants who advance to the next round of a contest.
 * The rule for advancing is that a participant's score must be positive and greater than or
 * equal to the score of the k-th place finisher.
 *
 * Algorithm:
 * 1. Read the total number of participants (n) and the rank of the cutoff score (k).
 * 2. Read the scores of all 'n' participants into an array.
 * 3. Iterate through the participants' scores and count how many have a score that is
 *    both positive and >= the score of the k-th participant.
 * 4. Print the final count.
 *
 * Time Complexity: O(N), where N is the number of participants, due to the linear scan of scores.
 * Space Complexity: O(N), to store the scores of the participants in an array.
 */
#include <iostream>

// n: total number of participants
// k: the rank of the participant establishing the score threshold
// i: index for reading scores into the array
// j: counter for participants who qualify for the next round
// a[51]: array to store the scores of up to 50 participants
int n, k, i, j, a[51];

int main() {
  // Read the number of participants and the k-th position.
  std::cin >> n >> k;
  
  /**
   * Block Logic: Reads all 'n' participant scores from standard input.
   * This compact while loop is equivalent to a for-loop that reads scores
   * into the array 'a'.
   */
  while (n > i)
    std::cin >> a[i++];
  
  /**
   * Block Logic: Counts the number of qualifying participants.
   * Invariant: The loop continues as long as the current participant's score
   * is positive (a[j] > 0) and greater than or equal to the k-th
   * participant's score (a[k-1]). The k-th score serves as the advancement threshold.
   */
  while (a[j] and a[j] >= a[k - 1])
    ++j;
  
  // Output the final count of advancing participants.
  std::cout << j;
}
