/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem from Codeforces.
 *
 * This program determines how many participants will advance to the next round of a contest.
 * A participant advances if their score is positive and greater than or equal to the
 * score of the k-th place finisher. The input scores are given in non-increasing order.
 *
 * @algorithm Linear Scan
 * @time_complexity O(N), where N is the number of participants, as the code iterates through the scores twice.
 * @space_complexity O(N) to store the scores (the array is fixed at 51, but effectively O(N)).
 */
#include <iostream>
using namespace std;

int main() {
  int arr[51]; // Array to store participant scores, size 51 for n_max=50.
  int n, k, result = 0;
  cin >> n >> k;

  /**
   * @brief Reads the scores of all n participants.
   * The problem guarantees these scores are in non-increasing order.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  k -= 1; // Adjust k to be a zero-based index.
  
  // Get the score of the k-th participant, which is the cutoff score.
  int val = arr[k];

  /**
   * @brief Counts the number of participants who qualify for the next round.
   *
   * A participant qualifies if their score is positive (arr[i] != 0) and
   * is greater than or equal to the cutoff score `val`.
   */
  for (int i = 0; i < n; i++) {
    if (arr[i] >= val && arr[i] != 0)
      result++;
  }
  cout << result;
  return 0;
}