/**
 * @file 112A.CPP
 * @brief This program performs a case-insensitive lexicographical comparison of two input strings.
 * It converts both strings to lowercase and then compares them to determine their relative order.
 * Algorithm: String manipulation (case conversion) followed by standard lexicographical comparison.
 * Time Complexity: O(L) where L is the maximum length of the input strings, due to character iteration.
 * Space Complexity: O(L) for creating lowercase copies of the strings.
 */
#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for string manipulation.

using namespace std;

/**
 * @brief Converts an input string to its lowercase equivalent.
 * @param s The string to be converted to lowercase.
 * @return The lowercase version of the input string.
 */
string toLowerCase(string s) {
  for (size_t i = 0; i < s.size(); ++i) {
    // Checks if the character is an uppercase letter (based on ASCII values)
    // and converts it to lowercase by adding 32.
    if (s[i] - 'a' < 0)
      s[i] += 32;
  }
  return s;
}

/**
 * @brief Compares two strings lexicographically, ignoring case.
 * @param first The first string for comparison.
 * @param second The second string for comparison.
 * @return -1 if 'first' is lexicographically smaller than 'second',
 *          1 if 'first' is lexicographically greater than 'second',
 *          0 if both strings are lexicographically equal.
 */
int compare(string first, string second) {
  // Converts both strings to lowercase for case-insensitive comparison.
  first = toLowerCase(first);
  second = toLowerCase(second);
  // Performs the lexicographical comparison.
  if (first < second)
    return -1;
  // If not smaller, checks for equality. If equal, returns 0, otherwise 'first' is greater.
  return first == second ? 0 : 1;
}

int main() {
  string first, second; // Declares two string variables to hold the input strings.
  // Reads the two strings from standard input.
  cin >> first >> second;
  // Calls the compare function and prints its result, followed by a newline.
  cout << compare(first, second) << endl;
  // Standard program exit.
  return 0;
}
