/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem.
 *
 * This program calculates the number of participants who advance to the next round of a competition.
 * Participants advance if their score is greater than or equal to the k-th place finisher's score,
 * and their own score is strictly positive.
 *
 * Algorithm: Iterative comparison with k-th score.
 * Time Complexity: O(N) for reading input and iterating through participants, where N is the number of contestants.
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to determine the number of advancing participants.
 *
 * Functional Utility: Handles input for number of contestants, k-th place, processes scores, and counts qualifiers.
 */
int main() {
  int contestant; // Total number of participants.
  int place;      // The k-th position whose score determines the cut-off.
  int counter = 0; // Counts the number of participants who advance.
  int score;      // Current participant's score.
  int lastscore;  // Score of the k-th placed participant, used as the threshold.

  // Block Logic: Reads the total number of contestants (n) and the k-th place (k).
  cin >> contestant;
  cin >> place;
  /**
   * Block Logic: Iterates through each participant's score, determining if they qualify.
   * Pre-condition: 'contestant' and 'place' (k) have been read.
   * Invariant: 'counter' accurately reflects the number of qualifiers encountered so far.
   */
  for (int i = 0; i < contestant; i++) {
    cin >> score;
    // Condition: If current participant is within the top 'place' ranks and has a positive score, they qualify.
    if (i < place && score > 0) {
      counter++;
      lastscore = score; // Update the score to beat for subsequent participants.
    }
    // Condition: If current participant's score matches the k-th place score and is positive, they qualify.
    // Note: The problem statement implies k-th score must be positive. If k-th score is 0, no one qualifies beyond that.
    else if (score == lastscore && score > 0) {
      counter++;
    }
    // Condition: If participant does not meet the criteria (score less than k-th, or zero score),
    // then no subsequent participant can qualify (as scores are typically non-increasing or this breaks the rule).
    else {
      break;
    }
  }
  // Output the total count of participants who advanced to the next round.
  cout << counter;
  return 0;
}