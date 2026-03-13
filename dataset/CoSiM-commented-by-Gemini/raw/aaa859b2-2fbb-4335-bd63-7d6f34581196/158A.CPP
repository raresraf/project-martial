/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces.
 * @details This program determines how many participants will advance to the next round.
 * The rule is that a participant advances if they have a positive score and their
 * score is greater than or equal to the score of the k-th place finisher.
 *
 * Algorithm:
 * 1. Read the total number of participants 'n' and the rank 'k'.
 * 2. Read all 'n' participant scores into an array 'pt'.
 * 3. The score to beat is the score of the k-th participant, which is at index 'k-1'.
 * 4. Iterate through all participants' scores.
 * 5. For each score, check if it is positive and greater than or equal to the k-th score.
 * 6. Increment a counter for each participant who meets these criteria.
 * 7. Print the final count.
 *
 * Time Complexity: O(N), for reading and then checking the scores.
 * Space Complexity: O(N), to store the scores in an array (specifically O(50) in this implementation).
 */
#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int n;      // Total number of participants.
  int k;      // The rank of the participant whose score is the benchmark.
  int pt[50]; // Array to store the scores of the participants.
  int count = 0; // Counter for participants who advance.

  // Read the number of participants and the benchmark rank.
  scanf("%d %d", &n, &k);

  // Read all participant scores into the array.
  for (int i = 0; i < n; i++)
    scanf("%d", &pt[i]);
  
  /**
   * Block Logic: Iterate through the scores to count who advances.
   * Invariant: 'count' holds the number of advancing participants from the first 'i' checked.
   */
  for (int i = 0; i < n; i++) {
    /**
     * NOTE: This block appears to be a flawed attempt to handle the edge case where
     * all scores are 0. If the first participant's score is 0, it incorrectly
     * checks the next participant's score, leading to a potential out-of-bounds
     * access on the last iteration. The correct logic is handled by the `else if`.
     */
    if (pt[0] == 0) {
      if (pt[i + 1] == 0)
        count = 0;
    } 
    // A participant advances if their score is >= the k-th participant's score
    // and their score is positive.
    else if (pt[i] >= pt[k - 1] && pt[i] != 0)
      count++;
  }

  // Print the total number of participants who advance.
  printf("%d\n", count);

  return 0;
}
