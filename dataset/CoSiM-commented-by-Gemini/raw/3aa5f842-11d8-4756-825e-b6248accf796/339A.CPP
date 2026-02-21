/**
 * @file 339A.CPP
 * @brief Rearranges a sum of numbers (1, 2, 3) into non-decreasing order.
 *
 * This program takes a string representing a sum of numbers (e.g., "3+2+1")
 * and sorts the numbers, printing the new sum (e.g., "1+2+3"). The input
 * numbers are guaranteed to be 1, 2, or 3.
 *
 * Algorithm:
 * The implementation uses a stable, counting-sort-like approach by making
 * multiple passes over the input string.
 * 1. Read the input string `s`.
 * 2. Create an empty result string `str`.
 * 3. First Pass: Iterate through `s`. For each character '1', append "1+" to `str`.
 * 4. Second Pass: Iterate through `s`. For each character '2', append "2+" to `str`.
 * 5. Third Pass: Iterate through `s`. For each character '3', append "3+" to `str`.
 * 6. After the three passes, `str` will contain the sorted numbers separated by
 *    '+', with an extra '+' at the end.
 * 7. Remove the trailing '+' character from `str`.
 * 8. Print the final, sorted string.
 *
 * Time Complexity: O(N), where N is the length of the input string. The program
 * makes three linear passes over the string.
 * Space Complexity: O(N), to store the resulting string `str`.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  string s, str;
  cin >> s;
  
  // First pass: collect all '1's.
  for (auto c : s) {
    if (c == '1') {
      str += c;
      str += '+';
    }
  }
  // Second pass: collect all '2's.
  for (auto c : s) {
    if (c == '2') {
      str += c;
      str += '+';
    }
  }
  // Third pass: collect all '3's.
  for (auto c : s) {
    if (c == '3') {
      str += c;
      str += '+';
    }
  }
  
  // Remove the trailing '+' from the constructed string.
  str.erase(str.length() - 1, 1);
  
  cout << str << endl;
  return 0;
}
