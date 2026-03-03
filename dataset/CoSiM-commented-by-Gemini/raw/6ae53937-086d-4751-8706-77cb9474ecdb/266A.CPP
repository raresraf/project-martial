
#include <iostream>
using namespace std;

/**
 * @file 266A.CPP
 * @brief Solution to Codeforces problem 266A, "Stones on the Table".
 * @details The program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color.
 *
 * Algorithm:
 * The program iterates through the row of stones, represented as a string. It
 * compares each stone with the one immediately next to it. If two adjacent
 * stones have the same color, it means one must be removed to satisfy the
 * condition. A counter is incremented for each such pair found.
 *
 * Time Complexity: O(N), where N is the number of stones, because it requires a
 * single pass through the string.
 * Space Complexity: O(N) to store the string of stone colors.
 */
int main() {
  // x: the total number of stones.
  int x;
  cin >> x;

  // col: a string representing the colors of the stones.
  string col;
  cin >> col;

  // coun: counter for the number of stones to be removed.
  int coun = 0;

  /**
   * @brief Count the number of adjacent stones with the same color.
   *
   * Pre-condition: `col` contains the sequence of stone colors.
   * Invariant: `coun` holds the number of removals needed for the scanned
   *            portion of the row.
   *
   * Logic: Iterate up to the second-to-last stone and compare each stone
   * `col[i]` with its neighbor `col[i+1]`. If they match, increment the counter.
   */
  for (int i = 0; i + 1 < x; i++) {
    if (col[i] == col[i + 1]) {
      coun++;
    }
  }

  // Output the total number of stones to remove.
  cout << coun;
  return 0;
}
