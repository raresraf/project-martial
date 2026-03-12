/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" problem from Codeforces.
 *
 * This program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color. The colors are
 * represented by characters in a string.
 *
 * @algorithm Linear Scan
 * @time_complexity O(N), where N is the number of stones, as it iterates through the string once.
 * @space_complexity O(N) to store the string of stone colors.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n, m = 0; // n: number of stones, m: count of stones to remove.
  string s;
  cin >> n;
  cin >> s;

  /**
   * @brief Iterates through the stones to count adjacent pairs of the same color.
   *
   * For each stone, it compares it with the next one. If they are the same,
   * it increments the counter `m`, as one of them must be removed.
   *
   * @note The loop condition `i < s.length()` allows `i` to reach `s.length() - 1`.
   * In this case, `s[i+1]` accesses `s[s.length()]`, which is out of bounds but
   * often evaluates to a null terminator in C++, usually not affecting the logic
   * but is technically incorrect. A safer condition would be `i < n - 1`.
   */
  for (int i = 0; i < s.length(); i++)
    m = s[i] == s[i + 1] ? m + 1 : m;
  
  cout << m << endl;
  return 0;
}