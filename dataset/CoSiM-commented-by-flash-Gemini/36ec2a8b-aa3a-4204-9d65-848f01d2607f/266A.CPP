/**
 * @file 266A.CPP
 * @brief Solution to a competitive programming problem: "Stones on the Table".
 *
 * This program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color. This is achieved by
 * counting the number of adjacent identical stones.
 *
 * Algorithm: String manipulation to remove adjacent duplicates and calculate the difference in length.
 * Time Complexity: O(L), where L is the length of the input string, due to single pass for unique and erase operations.
 */
#include <algorithm> // Functional Utility: Included for std::unique.
#include <cctype>    // Functional Utility: Included for character manipulation functions (though not directly used in final logic, often useful).
#include <cmath>     // Functional Utility: Included for mathematical functions (not directly used in final logic).
#include <cstring>   // Functional Utility: Included for C-style string functions (not directly used in final logic).
#include <iostream>  // Functional Utility: Included for standard input/output operations.
#include <string>    // Functional Utility: Included for std::string manipulation.
using namespace std; // Functional Utility: Uses the standard namespace to avoid prefixing standard library elements with std::

int main() {
  // Functional Utility: Optimizes standard I/O streams for faster execution,
  // crucial in competitive programming for time-sensitive tasks.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Variable Declarations:
  // n: The total number of stones (original length of the string).
  // s: The input string representing the sequence of stone colors.
  int n;
  cin >> n;
  string s;
  cin >> s;

  // Block Logic: Removes adjacent duplicate characters from the string.
  // std::unique: Rearranges the elements in the range [first, last) such that
  //              the first occurrence of each unique element is kept, and
  //              the order of the unique elements is preserved. It returns an
  //              iterator to the element that follows the last element not removed.
  // s.erase: Removes the elements from the position returned by unique to the end of the string.
  // The effect is that 's' now contains only unique adjacent characters.
  s.erase(unique(s.begin(), s.end()), s.end());

  // Output: Prints the number of stones that needed to be removed.
  // This is calculated as the original number of stones (n) minus the length
  // of the string after removing adjacent duplicates.
  cout << n - s.length() << endl;

  return 0; // Functional Utility: Indicates successful program execution.
}
