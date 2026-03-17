/**
 * @file 158A.CPP
 * @brief An attempted solution to the "Next Round" problem from Codeforces.
 *
 * @details The program is intended to find the number of participants who advance to the next round.
 * The logic implemented reads 'n' scores, identifies the score of the k-th participant as the cutoff,
 * and then attempts to determine the number of advancing players based on subsequent scores.
 *
 * @note The implementation appears to be a partial or incorrect solution to the standard "Next Round" problem.
 * The comments below describe the logic as it is written in the code.
 *
 * Time Complexity: O(n) for reading the inputs.
 * Space Complexity: O(n) to store the scores in a vector.
 */
// sumanth2303 - 16-11-2020
// problem link - https://codeforces.com/problemset/problem/158/A
// partially solved
#include <iostream>
using namespace std;
#include <bits/stdc++.h>

int main() {
  int n, k;
  cin >> n >> k;
  int a, score;
  vector<int> arr;
  // Block Logic: Reads n scores from standard input and stores them in a vector.
  for (int i = 0; i < n; i++) {
    cin >> a;
    arr.push_back(a);
  }
  
  // Functional Utility: Establishes the cutoff score based on the k-th participant's result.
  score = arr[k - 1];

  /**
   * @brief This block acts as a special case check.
   * Logic: If the first participant's score is 0, the program exits and prints 0,
   * assuming no one can advance.
   */
  if (arr[0] == 0) {
    cout << 0 << endl;
    return 0;
  }
  
  /**
   * @brief This loop iterates from the k-th participant to the end of the list.
   * Invariant: The loop's purpose is to check participants' scores starting from index 'k'.
   * The logic inside attempts to find the boundary where scores are no longer equal to the cutoff score.
   */
  for (int j = k; j < n; j++) {
    // This condition seems to handle the case where all 'n' participants are being considered up to k.
    if (n == k) {
      cout << k << endl;
      return 0;
    }
    // This condition checks if the score of the current participant 'j' differs from the cutoff score.
    if (score != arr[j] && score != 0) {
      cout << j << endl;
      return 0;
    }
  }

  // Default output if the loop completes without returning.
  cout << 0 << endl;
  return 0;
}