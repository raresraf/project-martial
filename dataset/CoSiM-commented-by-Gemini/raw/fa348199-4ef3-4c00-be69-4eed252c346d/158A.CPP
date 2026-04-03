/**
 * @file 158A.CPP
 * @author hardRocK
 * @date
 * @brief Solves the "Next Round" problem from Codeforces.
 * @see http://codeforces.com/problemset/problem/158/A
 *
 * @description
 * This program determines the number of participants who will advance to the next round of a competition.
 * The rule for advancing is that a participant's score must be equal to or greater than the score
 * of the k-th place finisher, and the participant must have a positive score.
 *
 * @algorithm
 * 1. Read the total number of participants (n) and the rank of the cutoff score (k).
 * 2. Read the scores of all n participants into an array. The scores are given in non-increasing order.
 * 3. Identify the score of the k-th participant. This score serves as the threshold.
 * 4. Iterate through all participants' scores.
 * 5. Count how many participants have a score that is greater than or equal to the threshold score
 *    and is also positive.
 * 6. Print the final count.
 *
 * @time_complexity O(N), where N is the number of participants, because the solution requires iterating through the list of scores twice.
 * @space_complexity O(N) to store the scores of the participants.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, k, val, cnt = 0;
  // Read the number of participants and the k-th position.
  cin >> n >> k;

  int arr[n];

  // Read all participant scores into an array.
  for (int i = 0; i < n; ++i) {
    cin >> arr[i];
  }

  // Determine the score of the k-th participant. This is the cutoff score.
  val = arr[k - 1];

  /**
   * @brief Count the number of participants who advance.
   * A participant advances if their score is positive and at least as high as the k-th participant's score.
   */
  for (int i = 0; i < n; ++i) {
    if (arr[i] >= val && arr[i] >= 1) {
      cnt++;
    }
  }

  // Output the total count of advancing participants.
  cout << cnt << endl;

  return 0;
}
