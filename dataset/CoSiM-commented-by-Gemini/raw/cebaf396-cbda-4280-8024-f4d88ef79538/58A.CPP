/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A, "Chat room".
 * @details The program checks if the word "hello" is a subsequence of a given
 *          input string.
 *
 * @algorithm
 * This solution uses a series of chained `string::find` calls to locate the
 * characters of "hello" in the correct order. It finds the first 'h', then
 * searches for an 'e' starting from the 'h's position, then for an 'l' after
 * the 'e', and so on. A final, verbose conditional statement checks if all
 * characters were found and if their indices are strictly increasing.
 *
 * @complexity
 * Time: O(N), where N is the length of the input string. Each `find` operation
 *       can scan the string, but since the starting points advance, the total
 *       work is bounded by a few passes over the string, resulting in linear time.
 * Space: O(N) to store the input string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;

  /**
   * @brief A chain of `string::find` calls to locate the characters of "hello"
   *        in sequential order within the input string `s`.
   *
   * h:  Finds the first 'h'.
   * e:  Finds the first 'e' at or after the position of 'h'.
   * l1: Finds the first 'l' at or after the position of 'e'.
   * l2: Finds the first 'l' starting one position *after* the first 'l' (l1).
   * o:  Finds the first 'o' at or after the position of the second 'l'.
   *
   * `string::find` returns string::npos (evaluates as -1 for signed int)
   * if the character is not found.
   */
  int h = s.find('h'), e = s.find('e', h), l1 = s.find('l', e),
      l2 = s.find('l', l1 + 1), o = s.find('o', l2);

  /**
   * @brief This condition explicitly checks two properties:
   *        1. All characters were found (their indices are not -1/npos).
   *        2. The indices of the found characters are strictly increasing,
   *           which confirms they form a valid subsequence.
   *
   * @note While verbose, this logic correctly verifies the subsequence. A
   *       simpler check like `o != -1` would also work because a failed find
   *       at any step would propagate `string::npos` to the final step `o`.
   */
  if (h < e && h < l1 && h < l2 && h < o && e < l1 && e < l2 && e < o &&
      l1 < l2 && l1 < o && l2 < o && h != -1 && e != -1 && l1 != -1 &&
      l2 != -1 && o != -1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
