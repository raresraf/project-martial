/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces.
 * @see https://codeforces.com/problemset/problem/158/A
 *
 * This program determines how many participants will advance to the next round
 * of a contest. A participant advances if their score is equal to or greater than
 * the k-th place finisher's score, and they have a positive score.
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n, k;
  int x[100];
  cin >> n >> k;

  int count = 0;
  for (int i = 0; i < n; i++) {
    cin >> x[i];
  }

  /**
   * @brief First pass: Count participants with scores greater than or equal to the k-th score.
   * The loop breaks early if a score is less than the k-th score, assuming the
   * scores are sorted in descending order.
   */
  for (int i = 0; i < n; i++) {

    if (x[i] >= x[k - 1]) {
      count++;
    } else {
      break;
    }
  }
  int count1 = 0;

  /**
   * @brief Second pass: Count participants from the previous count who have non-positive scores.
   * This two-pass approach is a bit convoluted; the check for a positive score
   * could be integrated into the first loop.
   */
  for (int i = 0; i < count; i++) {
    if (x[i] == 0 || x[i] < 0) {
      count1++;
    }
  }
  
  // The final result is the number of participants who met the score threshold
  // minus those with non-positive scores.
  cout << count - count1;
}
