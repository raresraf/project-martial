/**
 * @file 112A.CPP
 * @brief Lexicographical string comparison, ignoring case, using character iteration and `std::string::compare`.
 *
 * This program reads two strings, converts each character to lowercase using a loop,
 * and then compares them lexicographically using the `std::string::compare` method.
 * It outputs -1 if the first string is lexicographically smaller, 1 if it's larger,
 * and 0 if they are equal.
 *
 * Algorithm: Character iteration for case conversion followed by direct lexicographical comparison.
 * Time Complexity: O(L) where L is the length of the longer string, as both the
 *                  loop for `tolower` and `compare` operations are linear.
 * Space Complexity: O(1) beyond input storage, as `tolower` operates in-place on string characters.
 */
// 112A - Petya and Strings

#include <iostream> // Required for std::cin, std::cout.
#include <string>   // Required for std::string.
#include <cctype>   // Required for std::tolower.

using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: Reads two strings, converts them to lowercase, compares them lexicographically,
   *        and prints the comparison result (-1, 1, or 0).
   * @return 0 for successful execution.
   */
  string a, b; // Declare two strings for input.
  int i, l;    // 'i': loop counter, 'l': result of string comparison.

  cin >> a >> b; // Read the two input strings.

  /**
   * @brief Loop to convert all characters in both strings to lowercase.
   * @pre 'a' and 'b' contain the original input strings.
   * @post 'a' and 'b' contain their lowercase equivalents.
   * @invariant 'i' iterates from 0 to 'a.size() - 1'.
   */
  for (i = 0; i < a.size(); i++) {
    a[i] = char(tolower(a[i])); // Convert character in string 'a' to lowercase.
    b[i] = char(tolower(b[i])); // Convert character in string 'b' to lowercase.
  }

  // Functional Utility: Lexicographically compares 'a' and 'b'.
  // Returns <0 if 'a' is less than 'b', >0 if 'a' is greater than 'b', and 0 if they are equal.
  l = a.compare(b);

  /**
   * @brief Conditional block to print the comparison result based on 'l'.
   * @pre 'l' holds the result of the lexicographical comparison.
   */
  if (l < 0) // If 'a' is lexicographically smaller than 'b'.
    cout << "-1";
  else if (l > 0) // If 'a' is lexicographically greater than 'b'.
    cout << 1;
  else // If 'a' and 'b' are lexicographically equal.
    cout << 0;

  return 0; // Return 0 for successful execution.
}
