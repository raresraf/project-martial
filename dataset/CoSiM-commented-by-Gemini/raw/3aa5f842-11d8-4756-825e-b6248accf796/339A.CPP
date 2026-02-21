/**
 * @file 339A.CPP
 * @brief Solution to the "Helpful Maths" problem (Codeforces 339A).
 *
 * @details The problem requires sorting the numbers in a string that represents
 * a sum (e.g., "3+1+2") into non-decreasing order (e.g., "1+2+3"). The
 * numbers involved are only 1, 2, and 3.
 *
 * @algorithm This program employs a multi-pass sorting strategy, which is a
 * form of counting sort, tailored to the small set of numbers {1, 2, 3}.
 * 1. It makes a full pass through the input string to find and append all '1's
 *    to a result string, each followed by a '+'.
 * 2. It repeats this process in a second pass for all '2's.
 * 3. It repeats again in a third pass for all '3's.
 * This inherently builds the new string with the numbers sorted. Finally, the
 * superfluous trailing '+' is removed before printing the result.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string. The three
 *                  passes are each O(L), resulting in linear time complexity.
 * Space Complexity: O(L), to store both the original and the new result string.
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
