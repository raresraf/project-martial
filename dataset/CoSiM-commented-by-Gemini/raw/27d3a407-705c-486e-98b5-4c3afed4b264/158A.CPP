/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem (158A) from Codeforces.
 * @details This program determines how many participants will advance to the next round.
 * The condition for advancing is having a positive score that is greater than or equal
 * to the k-th place finisher's score.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution is a direct implementation of the problem statement.
 * 1. Read the number of participants (n) and the rank of the cutoff score (k).
 * 2. Read all participant scores into an array.
 * 3. The score at index `k-1` is the cutoff score.
 * 4. Iterate through the scores again, counting how many are both positive and greater
 *    than or equal to the cutoff score.
 * 5. Print the final count.
 *
 * @note This implementation uses some non-standard C++ features:
 * - `main()` is declared without a return type.
 * - A variable-length array (VLA) `int a[n]` is used, which is a C99 feature and a
 *   compiler extension in C++, not part of the C++ standard.
 *
 * @complexity
 * Time Complexity: O(N) - The program has two loops that each iterate up to N times.
 * Space Complexity: O(N) - An array of size N is used to store the scores.
 */
#include <iostream>
using namespace std;
// Using `main()` without an explicit `int` return type is non-standard in C++.
main() {
  int n, k, ctr = 0;
  cin >> n >> k;
  // A variable-length array (VLA) is a non-standard C++ extension.
  int a[n];
  /**
   * @block
   * @brief Reads all participant scores from standard input.
   * @pre 'n' holds the number of participants.
   * @post The array 'a' is populated with 'n' scores.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  /**
   * @block
   * @brief Counts participants who qualify for the next round.
   * @invariant 'ctr' holds the number of qualifying participants found so far.
   * @pre The array 'a' is populated, and 'k' is the 1-based cutoff rank.
   * @post 'ctr' holds the total count of qualifying participants.
   */
  for (int i = 0; i < n; i++) {
    // A participant qualifies if their score is positive and >= the k-th person's score.
    if (a[i] >= a[k - 1] && a[i] > 0) {
      ctr++;
    }
  }
  cout << ctr;
}
