/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces.
 * @details This program calculates the number of contestants who will advance to the
 * next round. A contestant advances if they have a positive score and their score
 * is greater than or equal to the k-th place finisher's score.
 * This implementation assumes the input scores are sorted in non-increasing order.
 *
 * Algorithm: Array traversal with early exit.
 * Time Complexity: O(N) in the worst case for reading and checking scores. However,
 * it can be faster if an early exit occurs.
 * Space Complexity: O(N) to store the contestant scores.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  int c = 0;

  int k, n, a[51], i;
  cin >> n >> k;

  /**
   * @brief Reads the scores of all contestants into an array.
   * Pre-condition: n contains the total number of contestants.
   */
  for (i = 0; i < n; i++) {
    cin >> a[i];
  }
  /**
   * @brief Counts the number of advancing contestants.
   * This loop iterates through the scores and counts contestants who qualify.
   * It relies on the input being sorted non-increasingly, allowing it to break
   * as soon as it finds a contestant who does not qualify.
   */
  for (i = 0; i < n; i++) {
    // A contestant advances if their score is positive and >= the k-th score.
    if (a[i] > 0 && a[i] >= a[k - 1]) {
      c++;
    } else {
      // Since scores are sorted, no further contestants can qualify.
      break;
    }
  }
  cout << c;
}

