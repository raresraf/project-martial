/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem.
 *
 * This program determines how many participants will advance to the next round of a contest.
 * The rule for advancing is that a participant's score must be positive and greater than
 * or equal to the score of the k-th place finisher.
 *
 * @algorithm Sorting and Linear Scan
 * @time_complexity O(N log N), dominated by the sorting step.
 * @space_complexity O(N) to store the scores of the participants.
 * @note The problem statement guarantees that the input scores are in non-increasing order,
 * so the sorting step is not strictly necessary for a compliant input. This implementation,
 * however, correctly handles unsorted input by sorting it first.
 */
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  // Dynamically allocate memory for scores.
  int *scores = new int[n];
  for (int u = 0; u < n; ++u) {
    cin >> scores[u];
  }

  // Sort the scores in ascending order.
  sort(scores, scores + n);

  // The k-th largest score is now at index n-k, which is the minimum score to advance.
  int min = scores[n - k];
  int promoted = 0;

  /**
   * @brief Counts the number of participants who advance to the next round.
   *
   * This loop iterates backward from the highest score in the sorted array.
   * It counts participants whose scores are positive and meet or exceed the k-th place score.
   *
   * Pre-condition: `scores` array is sorted in ascending order. `min` holds the score of the k-th finisher.
   * Invariant: `promoted` holds the count of qualifying participants found so far.
   */
  while (promoted < n && scores[n - 1 - promoted] > 0 &&
         scores[n - 1 - promoted] >= min) {
    ++promoted;
  }
  cout << promoted << endl;
  
  // It's good practice to free dynamically allocated memory, though often omitted in competitive programming.
  // delete[] scores;

  return 0;
}