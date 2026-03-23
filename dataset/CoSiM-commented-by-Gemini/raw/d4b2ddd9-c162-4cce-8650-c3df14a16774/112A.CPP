/**
 * @file 112A.CPP
 * @brief Solution to a string comparison problem, likely from a competitive programming contest.
 * @details The program performs a case-insensitive comparison of two input strings.
 * It reads two strings, converts both to lowercase, and then compares them lexicographically.
 * The output is -1 if the first string is lexicographically smaller than the second,
 * 1 if it is larger, and 0 if they are equal.
 *
 * @algorithm Lexicographical Comparison
 * @complexity
 * Time: O(L), where L is the length of the strings. The loop for converting to lowercase
 * and the subsequent comparison both take linear time.
 * Space: O(L), to store the two input strings.
 */
// 112A
#include <iostream>
#include <string> // It's better to include <string> for std::string
#include <cctype>   // For std::tolower

using namespace std;

int main() {
  string a, b;
  cin >> a >> b;

  /**
   * @brief Loop to convert both strings to lowercase.
   * Pre-condition: 'a' and 'b' contain the input strings.
   * Invariant: At the end of this loop, both strings will consist entirely of lowercase letters.
   */
  for (int i = 0; i < a.length(); i++) {
    // The use of magic numbers (65, 90, 32) is replaced by a more readable function.
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
  }

  /**
   * @brief Lexicographically compare the lowercase strings.
   * The comparison is done using the overloaded operators for std::string,
   * which perform a lexicographical comparison.
   */
  if (a < b)
    cout << "-1";
  else if (a > b)
    cout << "1";
  else // This is equivalent to `if (a == b)`
    cout << "0";

  return 0; // Indicate successful execution.
}