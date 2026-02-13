/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem.
 * @details The program determines how many participants will advance to the next round
 * of a competition. The rule is that a participant advances if their score is equal
 * to or greater than the score of the k-th place finisher, provided they also have a
 * positive score. The scores are given in non-increasing order.
 *
 * @section algorithm Algorithm
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read the `n` scores into an array. The scores are already sorted non-increasingly.
 * 3. Set an index `i` to `k-1`, pointing to the k-th participant. This participant's
 *    score is the initial cutoff score.
 * 4. **Handle Zero Score Case:** If the score of the k-th participant (`scores[i]`) is 0,
 *    no one with a 0 score can advance. The code then backtracks from `i` to find the
 *    index of the last participant who has a positive score.
 * 5. **Handle Positive Score Case:** If the k-th participant's score is positive, all
 *    participants with a score equal to the k-th finisher's score also advance. The code
 *    iterates forward from `i` to find the index of the last participant with this same score.
 * 6. The final answer is `i + 1`, representing the total number of participants who advance.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N) in the worst case, as the code may iterate through the list of scores once after the initial read.
 * - Space Complexity: O(N) to store the scores. (The code uses a fixed-size array of 50, sufficient for the problem constraints).
 */
#include <iostream>
using namespace std;

int main() {
  int n, k;
  int i = 0;
  int scores[50] = {0};
  cin >> n >> k;

  // Read the scores of all n participants.
  for (i = 0; i < n; i++) {
    cin >> scores[i];
  }

  // Set the initial index to the k-th participant.
  i = k - 1;

  // Case 1: The k-th participant scored 0. Since scores must be positive to
  // advance, we find the last participant (if any) who scored more than 0.
  if (scores[i] == 0) {
    // Backtrack from the k-th position to find the last non-zero score.
    while (i >= 0 && scores[i] == 0) {
      i--;
    }
  }
  // Case 2: The k-th participant has a positive score. All participants with
  // an equal score also advance.
  else {
    // Forward-track from the k-th position to include all participants with the same score.
    while (i < n - 1 && scores[i + 1] == scores[i]) {
      i++;
    }
  }
  // The index 'i' now points to the last person who qualifies.
  // 'i + 1' gives the total count of advancing participants.
  cout << i + 1;
}
