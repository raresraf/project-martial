/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces.
 * @details This program determines how many contestants will advance to the next round.
 * A contestant advances if their score is equal to or greater than the score of the
 * k-th place finisher, provided their score is positive.
 *
 * Algorithm: Array traversal and comparison.
 * Time Complexity: O(N) for reading the scores and O(N) for checking them, so O(N) overall.
 * Space Complexity: O(N) to store the scores of the contestants.
 */
#include <iostream>
using namespace std;
int main() {
  int n, k, count = 0;
  cin >> n >> k;
  int arr[n];
  /**
   * @brief Reads the scores of all contestants into an array.
   * Pre-condition: n contains the total number of contestants.
   * Invariant: The loop reads n integer scores from standard input.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }
  /**
   * @brief Counts the number of contestants who advance.
   * A contestant advances if their score is >= the k-th contestant's score
   * and their score is positive (not zero).
   */
  for (int i = 0; i < n; i++) {
    if (arr[i] >= arr[k - 1] && arr[i] != 0) {
      ++count;
    }
  }
  cout << count;
}

