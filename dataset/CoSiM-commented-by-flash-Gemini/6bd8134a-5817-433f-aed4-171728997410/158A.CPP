/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem: "Next Round".
 * Counts the number of participants who advance to the next round based on a scoring criterion.
 * Participants advance if their score is greater than or equal to the k-th placed participant's score and is positive.
 * Algorithm: Two-pass iteration. First to read scores, second to count qualifying participants.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(N) to store participant scores.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to solve the Next Round problem.
 * Reads the total number of participants 'n' and the k-th position 'k'.
 * Then reads 'n' scores, determines the k-th score, and counts participants meeting the criteria.
 */
int main() {
  int n, k, arr[102], cnt = 0, temp;
  cin >> n >> k;
  /**
   * Block Logic: Reads 'n' participant scores into an array.
   * Pre-condition: 'n' and 'k' have been read.
   * Invariant: After this loop, 'arr' contains all participant scores.
   */
  for (int i = 0; i < n; i++)
    cin >> arr[i];
  /**
   * Block Logic: Stores the score of the k-th placed participant.
   * Note: Array is 0-indexed, so k-th participant is at index k-1.
   */
  temp = arr[k - 1];

  /**
   * Block Logic: Iterates through all participants to count those who qualify.
   * Pre-condition: 'temp' holds the k-th participant's score.
   * Invariant: 'cnt' accumulates the number of qualifying participants.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Block Logic: Checks if a participant's score meets the advancement criteria.
     * Criteria: Score must be greater than or equal to 'temp' (k-th score) AND positive.
     */
    if (arr[i] >= temp && arr[i] > 0)
      cnt++;
  }
  cout << cnt << endl;
  return 0;
}

