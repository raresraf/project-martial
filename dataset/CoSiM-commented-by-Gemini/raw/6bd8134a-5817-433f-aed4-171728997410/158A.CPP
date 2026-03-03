
#include <bits/stdc++.h>
using namespace std;

/**
 * @file 158A.CPP
 * @brief Solution to Codeforces problem 158A, "Next Round".
 * @details The program counts the number of participants who qualify for the
 * next round of a competition.
 *
 * Algorithm:
 * A participant qualifies if they have a positive score that is greater than
 * or equal to the score of the k-th ranked participant. The program first reads
 * all participant scores into an array. It then stores the k-th participant's
 * score as a threshold value. Finally, it iterates through all scores again,
 * counting how many meet both criteria: being positive and being at least as
 * high as the threshold score.
 *
 * Time Complexity: O(N), for the two separate passes over the participant data
 * (one to read, one to count).
 * Space Complexity: O(N), to store the scores in an array.
 */
int main() {
  // n: number of participants, k: the rank to use as a benchmark.
  // arr: array to store scores.
  // cnt: counter for qualified participants.
  // temp: stores the threshold score from the k-th participant.
  int n, k, arr[102], cnt = 0, temp;

  cin >> n >> k;

  // Read all n scores into the array.
  for (int i = 0; i < n; i++)
    cin >> arr[i];

  // Store the score of the k-th participant (k-1 index) as the threshold.
  temp = arr[k - 1];

  /**
   * @brief Count the number of participants who advance.
   *
   * Pre-condition: Scores are in `arr`, and `temp` holds the threshold score.
   * Invariant: `cnt` holds the number of advancing participants found so far.
   */
  for (int i = 0; i < n; i++) {
    // A participant advances if their score is positive and >= the threshold.
    if (arr[i] >= temp && arr[i] > 0)
      cnt++;
  }

  // Output the final count.
  cout << cnt << endl;
  return 0;
}
