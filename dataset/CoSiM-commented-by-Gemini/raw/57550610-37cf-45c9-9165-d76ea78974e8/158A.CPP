/**
 * @file 158A.CPP
 * @brief An incorrect attempt to solve the "Next Round" problem from Codeforces.
 * @details The problem is to count participants who advance. A participant
 * advances if they have a positive score and their score is >= the k-th place
 * score. This implementation contains flawed logic for the main case.
 *
 * @algorithm (as implemented)
 * The code first finds the number of participants with positive scores.
 * 1. Read all scores into an array.
 * 2. Find the index of the first non-positive score and store it in `flag`.
 *    `flag` now effectively holds the count of participants with positive scores.
 * 3. The logic then splits into two main cases based on `k`.
 *    a. If `k > flag`: This means the k-th participant has a score of 0 or
 *       less. In this case, the code correctly prints `flag`, the number of
 *       people with positive scores (as they are the only ones who can advance).
 *    b. **Logical Flaw**: If `k <= flag`: This means the k-th participant has a
 *       positive score. The code then enters a `do-while` loop that seems to
 *       attempt to count participants with the same score as the k-th one,
 *       but its logic is incorrect. It fails to count participants ranked higher
 *       than `k` and does not correctly implement the problem's rules.
 *
 * @complexity
 * Time Complexity: O(n), due to the loops through the scores.
 * Space Complexity: O(n) for the scores array.
 */
#include <iostream>

using namespace std;

int main() {
  int n, k;
  cin >> n >> k; // n participants, k-th place cutoff

  // Using a Variable Length Array (VLA), a non-standard C++ extension.
  int scores[n];
  // `flag` will store the index of the first non-positive score.
  // Initializing to a value > n would be safer, but this works if all scores are positive.
  int flag = n;

  // input
  for (int i = 0; i < n; i++) {
    cin >> scores[i];
  }

  /**
   * @block
   * @brief Find the index of the first non-positive score.
   * After this loop, `flag` effectively represents the count of participants
   * with positive scores.
   */
  for (int i = 0; i < n; i++) {
    if (scores[i] <= 0) {
      flag = i;
      break;
    }
  }

  // The logic is split based on whether the k-th rank is within the group
  // of positive-scoring participants.
  if (k <= flag) {
    // FLAWED LOGIC BLOCK:
    // This part of the code is incorrect for the problem statement.
    // It appears to try to find subsequent contestants with the same score as
    // the k-th one, but it fails to count all participants from 1 to k-1 who
    // should also advance and doesn't handle all conditions correctly.
    int i = 0;
    do {
      if ((k + i) < n && scores[k + i] == scores[k - 1])
        i++;
      else
        break;
    } while (k + i < n);
    cout << k + i;
  } else {
    // This case is correct: if the k-th person's rank is beyond the last
    // person with a positive score, then only people with positive scores advance.
    // 'flag' correctly holds this count.
    cout << flag;
  }
  return 0;
}
