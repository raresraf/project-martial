/**
 * @file 112A.CPP
 * @brief Lexicographical comparison of two strings after case normalization.
 * 
 * This program reads two strings, converts them to lowercase, and then
 * compares them lexicographically. It prints -1 if the first string is
 * smaller, 1 if the second string is smaller, and 0 if they are equal.
 * This is a common task in competitive programming.
 * 
 * @algorithm Case normalization followed by direct lexicographical comparison.
 * @time_complexity O(L), where L is the length of the strings. This is because
 *                  the program iterates through the strings once for conversion
 *                  and once for comparison.
 * @space_complexity O(L) to store the two input strings.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Converts a given string to lowercase.
 * 
 * This function iterates over each character of the input string and
 * converts any uppercase letters (A-Z) to their lowercase equivalents
 * by adding 32 to their ASCII value.
 * 
 * @param w The input string to be converted.
 * @return The lowercase version of the input string.
 */
string toLower(string w) {
  for (int i = 0; i < w.length(); i++)
    if (w[i] >= 'A' && w[i] <= 'Z')
      w[i] += 32;
  return w;
}

/**
 * @brief Main function to read, process, and compare the strings.
 */
int main() {
  string a, b;
  // Read the two strings from standard input.
  cin >> a >> b;

  // Normalize both strings to lowercase for case-insensitive comparison.
  a = toLower(a);
  b = toLower(b);

  /**
   * @brief Perform lexicographical comparison and print the result.
   * 
   * The core logic relies on the overloaded '<' and '==' operators for
   * std::string, which perform a character-by-character comparison.
   * - Prints "-1" if string 'a' comes before string 'b'.
   * - Prints "1" if string 'b' comes before string 'a'.
   * - Prints "0" if they are identical.
   */
  if (a < b)
    printf("-1");
  if (b < a)
    printf("1");
  if (a == b)
    printf("0");
  return 0;
}