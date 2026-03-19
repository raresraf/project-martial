/**
 * @file 112A.CPP
 * @brief Solution to Codeforces Problem 112A: "Petya and Strings".
 *
 * This problem asks to compare two strings lexicographically, ignoring case.
 * The output should be -1 if the first string is lexicographically smaller,
 * 1 if it is larger, and 0 if they are equal.
 *
 * Algorithm: Read two input strings. Convert them to C-style strings and use
 * `strcasecmp` (a non-standard C library function, typically a GNU extension)
 * to perform a case-insensitive lexicographical comparison. The return value
 * of `strcasecmp` directly matches the required output.
 *
 * Time Complexity: O(L), where L is the maximum length of the input strings,
 * due to the linear scan performed by `strcasecmp`.
 * Space Complexity: O(L) to store the input strings.
 */
#include <iostream> // Required for `std::cin` and `std::cout`.
#include <string>   // Required for `std::string`.

#include <cstring> // Required for `strcasecmp`. Note: `strcasecmp` is a non-standard function.

/**
 * @brief Helper function to print an integer followed by a newline.
 * @param v The integer value to print.
 */
void answer(int v) { std::cout << v << '\n'; }

/**
 * @brief Solves the "Petya and Strings" problem for a given pair of strings.
 * Performs a case-insensitive lexicographical comparison of the two strings.
 *
 * @param s1 The first string to compare.
 * @param s2 The second string to compare.
 */
void solve(const std::string &s1, const std::string &s2) {
  // Uses `strcasecmp` to compare the C-style representations of the strings,
  // ignoring case. `strcasecmp` returns:
  // - a negative value if `s1` < `s2`
  // - a positive value if `s1` > `s2`
  // - 0 if `s1` == `s2`
  // This directly matches the problem's output requirements.
  answer(strcasecmp(s1.c_str(), s2.c_str()));
}

/**
 * @brief Main function of the program.
 * Reads two strings from standard input and calls `solve` to compare them.
 */
int main() {
  std::string s1, s2; // Declare two `std::string` objects to store input.
  std::cin >> s1 >> s2; // Reads the two strings from standard input.

  solve(s1, s2); // Calls the solve function to compare the strings and print the result.

  return 0; // Indicates successful execution.
}
