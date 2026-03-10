/**
 * @file 266A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Stones on the Table").
 * This program calculates the minimum number of stones to remove from a row
 * of colored stones such that no two adjacent stones have the same color.
 *
 * @algorithm Linear Scan and Counting. The algorithm iterates through the stones
 * and counts every time a stone is the same color as the one immediately
 * preceding it.
 *
 * @time_complexity O(N), where N is the number of stones, due to a single pass.
 * @space_complexity O(N) to store the string representing the stone colors.
 */
#include <iostream>
using namespace std;
int main() {
  int n;           // The total number of stones.
  int count = 0;   // The number of stones to remove.
  string s;        // The sequence of stone colors.
  cin >> n >> s;

  /**
   * @block
   * @brief Counts the number of adjacent identical stones.
   *
   * The loop iterates from the second stone to the end, comparing each stone `s[i]`
   * with its predecessor `s[i-1]`. Each time a pair of identical adjacent stones
   * is found, it means one must be removed, so the counter is incremented.
   */
  for (int i = 1; i < n; i++) {
    if (s[i] == s[i - 1])
      count++;
  }
  cout << count << endl;
  return 0;
}
