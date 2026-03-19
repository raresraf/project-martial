/**
 * @file 112A.CPP
 * @brief Solution to Codeforces Problem 112A: "Petya and Strings".
 * Compares two input strings lexicographically, ignoring case, and prints -1, 1, or 0.
 * Algorithm: Converts both strings to lowercase and then performs a standard lexicographical comparison.
 * Time Complexity: O(L_max), where L_max is the length of the longer string, due to string iteration and comparison.
 * Space Complexity: O(L_max) to store the input strings and their lowercase versions.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++, including iostream and string.

using namespace std;

/**
 * @brief Converts an input string to its lowercase equivalent.
 * Iterates through the string and converts each uppercase character to lowercase
 * by adding 32 to its ASCII value.
 *
 * @param w The input string to be converted to lowercase.
 * @return The lowercase version of the input string.
 */
string toLower(string w) {
  /**
   * Block Logic: Iterates through each character of the string.
   * Pre-condition: 'w' is a valid string.
   * Invariant: Characters from index 0 to i-1 have been processed.
   */
  for (int i = 0; i < w.length(); i++) {
    /**
     * Block Logic: Checks if the current character is an uppercase letter.
     * If true, converts it to lowercase.
     */
    if (w[i] >= 'A' && w[i] <= 'Z') {
      w[i] += 32; // ASCII value of 'a' is 32 greater than 'A'.
    }
  }
  return w;
}

/**
 * @brief Main function to solve the "Petya and Strings" problem.
 * Reads two strings, converts them to lowercase, and compares them lexicographically.
 */
int main() {
  string a, b; // Declare two strings to store input.
  cin >> a >> b; // Read the two strings from standard input.
  
  // Convert both strings to lowercase for case-insensitive comparison.
  a = toLower(a);
  b = toLower(b);
  
  /**
   * Block Logic: Performs lexicographical comparison of the two lowercase strings.
   * Prints -1 if 'a' is lexicographically smaller than 'b'.
   */
  if (a < b) {
    printf("-1");
  }
  /**
   * Block Logic: Prints 1 if 'b' is lexicographically smaller than 'a' (meaning 'a' is larger).
   */
  if (b < a) {
    printf("1");
  }
  /**
   * Block Logic: Prints 0 if the two strings are lexicographically equal.
   */
  if (a == b) {
    printf("0");
  }
  return 0;
}