/**
 * @file 158A.CPP
 * @brief Counts participants advancing to the next round based on a score threshold.
 *
 * This program determines how many participants will advance based on their scores.
 * The rule for advancing is that a participant's score must be positive and greater
 * than or equal to the score of the k-th place finisher.
 *
 * @algorithm Threshold-based counting.
 * @time_complexity O(N), where N is the number of participants, due to two linear scans of the scores.
 * @space_complexity O(N) to store the scores of the participants in a dynamic array.
 */
#include <iostream>
using namespace std;

int main() {
  // n: number of participants, k: the rank to determine the score threshold.
  int n, k;
  cin >> n >> k;

  // Dynamically allocate an array to store the scores of n participants.
  int *a = new int[n];
  int i;

  /**
   * @brief Block Logic: Reads all participant scores from standard input.
   * @invariant After this loop, the array `a` is populated with `n` scores.
   */
  for (i = 0; i < n; i++)
    cin >> a[i];

  // @brief The score of the k-th participant (0-indexed), which serves as the threshold.
  int bar = a[k - 1];
  // @brief Counter for the number of participants who will advance.
  int c = 0;

  /**
   * @brief Block Logic: Counts the participants who meet the advancement criteria.
   *
   * This loop iterates through all participants and increments a counter if
   * a participant's score is positive and meets or exceeds the threshold score.
   * @invariant `c` holds the count of participants who have passed the criteria so far.
   */
  for (i = 0; i < n; i++) {
    // A participant advances if their score is positive AND >= the k-th place score.
    if (a[i] > 0 && a[i] >= bar)
      ++c;
  }

  // Output the final count of advanced participants.
  cout << c << endl;

  // Functional Utility: Deallocates the memory used for the scores array to prevent memory leaks.
  delete a;
  a = NULL;

  return 0;
}
