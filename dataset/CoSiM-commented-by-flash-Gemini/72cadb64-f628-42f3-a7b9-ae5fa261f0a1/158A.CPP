/**
 * @file 158A.CPP
 * @brief Solution to Codeforces Problem 158A: "Next Round".
 * Counts the number of participants who will advance to the next round of a competition.
 * Participants advance if their score is positive and greater than or equal to the score
 * of the participant in the k-th place.
 * Algorithm: Read all scores, identify the k-th score as the threshold, then iterate
 * through all scores to count those meeting the criteria. This solution assumes scores
 * are provided in non-increasing order, allowing direct identification of the k-th score.
 * Time Complexity: O(N), where N is the number of participants, due to two linear passes
 *                  for input and counting.
 * Space Complexity: O(N) to store participant scores.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++, such as iostream for cin/cout.
using namespace std;

/**
 * @brief Main function to solve the "Next Round" problem.
 * Reads participant scores, determines the k-th place score, and counts qualifiers.
 */
int main() {
  // Declare variables:
  // 'n': total number of participants.
  // 'k': the rank of the participant whose score sets the passing threshold.
  // 'a[105]': array to store the scores of participants (using 1-based indexing in loops).
  // 'b': stores the score of the k-th place participant.
  // 'cn': counter for the number of participants who advance to the next round.
  int n, k, a[105], b, cn = 0;
  cin >> n >> k; // Reads the total number of participants (n) and the k-th place (k).

  /**
   * Block Logic: Reads 'n' scores from input and stores them in the array 'a', using 1-based indexing.
   * Pre-condition: 'n' specifies the exact number of scores to read.
   * Invariant: After each iteration, 'a[1]' through 'a[i]' have been read.
   */
  for (int i = 1; i <= n; ++i) {
    cin >> a[i]; // Reads the score of the i-th participant.
  }
  
  // Determines the minimum score required to pass.
  // The problem statement implies scores are given in non-increasing order, so a[k] is the k-th score.
  b = a[k];

  /**
   * Block Logic: Iterates through all participant scores to count those who qualify.
   * Qualification Criteria: A participant advances if their score is positive AND
   *                         is greater than or equal to the k-th place score ('b').
   * Invariant: After each iteration, 'cn' accurately counts the qualified participants found so far.
   */
  for (int i = 1; i <= n; ++i) {
    if (a[i] >= b && a[i] > 0) {
      ++cn; // Increment the counter if the participant qualifies.
    }
  }
  
  cout << cn << endl; // Prints the total number of participants who advance, followed by a newline.
}
