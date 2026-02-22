/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A - Helpful Maths.
 *
 * @algorithm
 * The problem requires sorting the numbers in a string that represents a sum
 * (e.g., "3+1+2" should become "1+2+3"). The numbers are guaranteed to be
 * 1, 2, or 3.
 *
 * This solution uses a "Counting Sort" approach, which is highly efficient
 * given the small and fixed range of input numbers.
 *
 * The algorithm is as follows:
 * 1. Read the input string `s`.
 * 2. Iterate through the input string once, counting the occurrences of '1', '2',
 *    and '3'. Store these in `count1`, `count2`, and `count3`.
 * 3. Build a new result string `r`.
 * 4. Append `count1` instances of "1+", then `count2` instances of "2+", and
 *    finally `count3` instances of "3+" to the result string `r`.
 * 5. This construction will leave a trailing '+' at the end of the string `r`.
 * 6. Print the result string `r` up to the second-to-last character to exclude
 *    the final extraneous '+'.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the string. The process
 *   involves a few separate passes through the string/counts, but each is linear.
 * - Space Complexity: O(L), to store the original and result strings.
 */
// 339A
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s, r;
  cin >> s;
  int count1 = 0, count2 = 0, count3 = 0;
  /**
   * @brief Stage 1: Count the occurrences of each number (1, 2, 3).
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1') {
      count1++;
    } else if (s[i] == '2') {
      count2++;
    } else if (s[i] == '3') {
      count3++;
    }
  }
  /**
   * @brief Stage 2: Rebuild the string `r` in sorted order.
   * Appends the numbers and '+' separators based on the counts.
   */
  for (int i = 0; i < count1; i++) {
    r.push_back('1');
    r.push_back('+');
  }
  for (int i = 0; i < count2; i++) {
    r.push_back('2');
    r.push_back('+');
  }
  for (int i = 0; i < count3; i++) {
    r.push_back('3');
    r.push_back('+');
  }
  /**
   * @brief Stage 3: Print the result.
   * The loop iterates up to `r.length() - 1` to avoid printing the
   * trailing '+' that was added in the rebuilding stage.
   */
  for (int i = 0; i < r.length() - 1; i++)
    cout << r[i];
  return 0;
}
