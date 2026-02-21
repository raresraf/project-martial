/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings, converts them both to lowercase, and then
 * compares them lexicographically. It prints 0 if they are equal, 1 if the
 * first string is greater, and -1 if the first string is smaller.
 *
 * Algorithm:
 * 1. Read two strings, `s1` and `s2`.
 * 2. Iterate through both strings from the first to the last character. In each
 *    iteration, convert the character at the current index to lowercase for
 *    both strings.
 * 3. After the loop, both strings are entirely in lowercase.
 * 4. Use the standard C++ string comparison operators (`==`, `>`, `<`) to
 *    perform a lexicographical comparison on the modified strings.
 * 5. Store the result (0, 1, or -1) in an integer variable.
 * 6. Print the result.
 *
 * Time Complexity: O(N), where N is the length of the strings. The program
 * makes one pass to convert to lowercase (O(N)) and another pass for the
 * final comparison (also O(N) in the worst case).
 * Space Complexity: O(N) to store the two input strings.
 */
#include <iostream>
#include <string> // Recommended for std::string operations
#include <cctype>   // Recommended for tolower

using namespace std;

int main() {
  string s1 = "";
  string s2 = "";
  int result = 0;
  cin >> s1 >> s2;
  
  /**
   * @brief Convert both strings to lowercase in-place.
   * This ensures the subsequent comparison is case-insensitive.
   */
  for (int i = 0; i < s1.length(); i++) {
    s1[i] = tolower(s1[i]);
    s2[i] = tolower(s2[i]);
  }

  /**
   * @brief Compare the lowercase strings lexicographically.
   * The overloaded operators for std::string handle the comparison.
   */
  if (s1 == s2) {
    result = 0;
  } else if (s1 > s2) {
    result = 1;
  } else if (s1 < s2) {
    result = -1;
  }

  cout << result << endl;

  return 0;
}
