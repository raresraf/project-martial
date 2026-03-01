/**
 * @file 339A.CPP
 * @brief Solution for the "Helpful Maths" problem.
 *
 * @problem Given a string representing a sum of single-digit numbers (1, 2, or 3),
 * for example "3+2+1", the task is to rearrange the numbers in non-decreasing
 * order and print the new sum, e.g., "1+2+3".
 *
 * @algorithm The code solves this using direct string manipulation in three phases:
 * 1. Filter: It iterates through the string to remove all '+' characters,
 *    leaving a string containing only the digits.
 * 2. Sort: It sorts the string of digits. Since characters '1', '2', '3' have
 *    ascending ASCII values, this correctly sorts the numbers.
 * 3. Reconstruct: It iterates through the sorted digit string and re-inserts
 *    '+' characters between them to form the final sorted expression.
 *
 * @time_complexity O(N^2), where N is the length of the string. The filtering
 * and reconstruction loops have this complexity because `string::erase` and
 * `string::insert` can take linear time, and they are inside a loop.
 * @space_complexity O(N) to store the string.
 */
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  cin >> s;

  // 1. Filter Phase: Remove all '+' characters.
  // Note: This loop is fragile. Modifying a collection while iterating over it
  // this way can lead to bugs, but it works for this specific problem format.
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '+') {
      s.erase(s.begin() + i);
    }
  }

  // 2. Sort Phase: Sort the remaining characters (which are digits).
  sort(s.begin(), s.end());

  int len = s.length();
  int it = 1;
  // 3. Reconstruct Phase: Re-insert the '+' characters.
  for (int i = 0; i < len - 1; i++) {
    s.insert(it, "+");
    it += 2; // Move the insertion point past the digit and the new '+'.
  }
  
  cout << s;
  return 0;
}
