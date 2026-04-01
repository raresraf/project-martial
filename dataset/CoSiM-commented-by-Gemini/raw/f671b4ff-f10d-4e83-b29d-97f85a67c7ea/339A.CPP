/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces (339A).
 * @details The problem requires rearranging a given sum of numbers (containing only 1, 2, and 3)
 * into non-decreasing order. For example, "3+2+1" should become "1+2+3".
 *
 * Algorithm: Counting Sort.
 * 1. Read the input string representing the sum.
 * 2. Count the occurrences of each number ('1', '2', '3').
 * 3. Reconstruct the string by placing the numbers in sorted order, separated by '+'.
 *    This implementation does this in-place.
 *
 * Time Complexity: O(L) where L is the length of the string, for two linear passes.
 * Space Complexity: O(1) besides the storage for the input string itself.
 */
#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string s;
  int c1 = 0, c2 = 0, c3 = 0, i;
  cin >> s;
  /**
   * @brief First pass: Count the occurrences of each number.
   * This loop iterates through the string and increments counters for '1', '2', and '3'.
   */
  for (i = 0; i < s.length(); i++) {
    if (s[i] == '1') {
      c1++;
    }
    if (s[i] == '2') {
      c2++;
    }
    if (s[i] == '3') {
      c3++;
    }
  }
  /**
   * @brief Second pass: Rebuild the string in sorted order.
   * This loop iterates through the positions where numbers should be (every second character)
   * and fills them in based on the counts, starting with '1's, then '2's, then '3's.
   */
  for (i = 0; i < s.length(); i += 2) {
    if (c1 > 0) {
      s[i] = '1';
      c1--;
    } else if (c2 > 0) {
      s[i] = '2';
      c2--;
    } else if (c3 > 0) {
      s[i] = '3';
      c3--;
    }
  }
  cout << s << endl;
  return 0;
}

