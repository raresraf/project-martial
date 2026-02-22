/**
 * @file 158A.CPP
 * @brief Solution to Codeforces problem 158A - Next Round.
 *
 * @algorithm
 * The problem is to count how many participants advance to the next round. The
 * rule is that a participant's score must be positive and greater than or equal
 * to the score of the k-th ranked participant. The input scores are guaranteed
 * to be in non-increasing order.
 *
 * This solution leverages the non-increasing order of scores for an efficient
 * single-pass check after an initial read.
 *
 * The algorithm is as follows:
 * 1. Read the total number of participants `n` and the rank `k`.
 * 2. Read all `n` scores into an array `a`.
 * 3. Get the score of the k-th participant, `val = a[k-1]`.
 * 4. Use a single `while` loop to iterate from the beginning of the array.
 * 5. Continue incrementing a counter `i` as long as the score `a[i]` meets the
 *    advancement criteria (`a[i] >= val` and `a[i] > 0`).
 * 6. Because the scores are non-increasing, the first participant who does not
 *    qualify marks the end of the list of qualifiers. The loop will terminate.
 * 7. The final value of the counter `i` is the number of participants who advance.
 *
 * @note The code uses `int a[n]`, a variable-length array (VLA), which is a
 *       non-standard C++ extension.
 *
 * @complexity
 * - Time Complexity: O(N), for reading the input and then the single pass to count.
 * - Space Complexity: O(N), to store the `n` participant scores.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, i;
  cin >> n >> k;
  int a[n]; // VLA, a non-standard C++ extension.

  // Read all scores into the array.
  for (i = 0; i < n; i++) {
    cin >> a[i];
  }
  
  // The score to beat is that of the k-th participant (0-indexed).
  int val = a[k - 1];
  i = 0; // Reset `i` to use as a counter for qualified participants.

  /**
   * @brief Counts the number of participants who qualify.
   * The loop continues as long as the participant's score is positive and
   * greater than or equal to the k-th score. Since the array is sorted in
   * non-increasing order, we can stop as soon as we find the first person
   * who doesn't qualify.
   */
  while (a[i] >= val && a[i] > 0) {
    i++;
    // Safety break in case all `n` participants qualify.
    if (i == n) {
      break;
    }
  }
  // The final value of `i` is the count of qualified participants.
  cout << i;
}
