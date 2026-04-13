// Petya and Strings
// https://codeforces.com/problemset/problem/112/A
//
// Little Petya loves presents. His mum bought him two strings of the same size
// for his birthday. The strings consist of uppercase and lowercase Latin
// letters. Now Petya wants to compare those two strings lexicographically. The
// letters' case does not matter, that is an uppercase letter is considered
// equivalent to the corresponding lowercase letter. Help Petya perform the
// comparison.
//
// Input
// Each of the first two lines contains a bought string. The strings' lengths
// range from 1 to 100 inclusive. It is guaranteed that the strings are of the
// same length and also consist of uppercase and lowercase Latin letters.
//
// Output
// If the first string is less than the second one, print "-1". If the second
// string is less than the first one, print "1". If the strings are equal, print
// "0". Note that the letters' case is not taken into consideration when the
// strings are compared.

/**
 * @file 112A.CPP
 * @brief Solution to a competitive programming problem: "Petya and Strings".
 *
 * This program performs a case-insensitive lexicographical comparison between two input strings
 * of equal length. It converts both strings to lowercase before comparison to ensure
 * case-insensitivity. The output indicates whether the first string is lexicographically
 * smaller (-1), larger (1), or equal (0) to the second string.
 *
 * Algorithm: Character-by-character comparison after uniform case conversion.
 * Time Complexity: O(L), where L is the length of the input strings. This involves one
 *                  pass to convert characters to lowercase and another pass for comparison.
 * Space Complexity: O(L) to store the input strings.
 */
#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for using std::string and its functionalities.
// Required for tolower function. <cctype> is more appropriate but <iostream> might pull it.
#include <cctype> // Explicitly include <cctype> for tolower.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

/**
 * @brief Main function: entry point of the program.
 * Reads two strings, converts them to lowercase, and then performs a
 * lexicographical comparison to determine their relative order.
 */
int main() {

  string str1, str2; // Declares two string variables to store the input.
  int i;             // Loop counter for comparison.
  cin >> str1;       // Reads the first string from standard input.
  cin >> str2;       // Reads the second string from standard input.
  int len = str1.length(); // Gets the length of the strings (guaranteed to be equal).

  /**
   * @brief Converts both strings to lowercase for case-insensitive comparison.
   * Block Logic: Iterates through each character of both strings and converts
   * them to their lowercase equivalents using `tolower()`.
   * Pre-condition: `str1` and `str2` contain the original input strings.
   */
  for (int i = 0; i < len; i++) {
    str1[i] = tolower(str1[i]); // Inline: Converts the character at index `i` in `str1` to lowercase.
    str2[i] = tolower(str2[i]); // Inline: Converts the character at index `i` in `str2` to lowercase.
  }

  /**
   * @brief Performs character-by-character lexicographical comparison.
   * Block Logic: Iterates through the strings. At the first differing character,
   * it determines which string is lexicographically smaller or larger and prints
   * the corresponding result (-1 or 1). The loop terminates upon finding a difference.
   * Pre-condition: Both strings are of the same length and have been converted to lowercase.
   */
  for (i = 0; i < str1.length(); i++) {
    if (str1[i] - str2[i] > 0) { // If character in str1 is greater than in str2.
      cout << 1 << endl;         // str1 is lexicographically larger.
      break;                     // Exit the loop as comparison result is found.
    } else if (str1[i] - str2[i] < 0) { // If character in str1 is less than in str2.
      cout << -1 << endl;          // str1 is lexicographically smaller.
      break;                       // Exit the loop.
    } else
      continue; // Characters are equal, continue to the next character.
  }

  /**
   * @brief Checks if strings are equal.
   * Block Logic: If the loop completed without a `break` (meaning `i` reached `str1.length()`),
   * it implies all characters were identical, so the strings are equal.
   */
  if (i == str1.length())
    cout << 0 << endl; // Prints 0 if strings are lexicographically equal.

  return 0; // Indicates successful program execution.
}
