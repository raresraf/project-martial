/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem involving string abbreviation.
 *
 * This program reads a list of words and abbreviates them if their length
 * exceeds 10 characters. Abbreviation involves replacing the middle characters
 * with the count of those characters.
 *
 * Algorithm: String length check and abbreviation.
 * Time Complexity: O(N * L), where N is the number of strings and L is the maximum length of a string.
 */
#include <iostream>
#include <string.h>

using namespace std;

int main() {
  // Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout
  // to optimize high-throughput I/O required for competitive programming constraints.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n, len;
  char str[100][101];

  // Block Logic: Reads the number of strings to be processed.
  cin >> n;

  // Block Logic: Reads 'n' strings from standard input into a 2D character array.
  // Invariant: Each row 'i' of 'str' will contain one input string.
  for (int i = 0; i < n; ++i) {
    cin >> str[i];
  }

  // Block Logic: Iterates through each string to apply abbreviation logic.
  // Invariant: For each iteration, `str[i]` holds the current string to be evaluated.
  for (int i = 0; i < n; ++i) {

    // Block Logic: Calculates the number of characters to be replaced in the abbreviation.
    // len represents the count of characters between the first and last characters (exclusive).
    len = strlen(str[i]) - 2;

    // Block Logic: Checks if the string length is within the non-abbreviation limit (<= 10 characters).
    // Precondition: `len` is the count of characters excluding the first and last.
    if (len <= 8) { // Original length is len + 2. So, if len <= 8, original length is <= 10.
      // Output: Prints the original string if its length is 10 or less.
      cout << str[i] << endl;

    } else {
      // Output: Prints the abbreviated string.
      // The format is: first character, followed by the count of middle characters,
      // followed by the last character.
      cout << str[i][0] << len << str[i][len + 1] << endl;
    }
  }

  return 0;
}