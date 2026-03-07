/**
 * @file 58A.CPP
 * @brief Solution to a competitive programming problem that checks if the word "hello"
 * can be formed as a subsequence from a given input string.
 * A subsequence is a sequence that can be derived from another sequence by deleting
 * some or no elements without changing the order of the remaining elements.
 * Algorithm: Iterative string parsing and character matching to build the "hello" subsequence.
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(1) for helper strings (constant size "hello").
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

/**
 * @brief Checks if a given character `s` appears within a string `str`.
 *
 * @param s The character to search for.
 * @param str The string to search within.
 * @return bool: True if the character is found, False otherwise.
 */
bool IfCharInString(char s, string str) {
  bool appear = false; // Flag to indicate if the character is found.

  /**
   * Block Logic: Iterates through the characters of the string `str`.
   * Invariant: At the start of each iteration, 'i' is the current index.
   */
  for (int i = 0; i < str.size(); i++) {
    if (str[i] == s) { // Conditional Logic: If the character matches.
      appear = true;   // Set flag to true.
      break;           // Exit loop early as character is found.
    }
  }
  return appear;
}

/**
 * @brief Extracts characters from an input string that form the word "hello" as a subsequence.
 * This function attempts to build the word "hello" sequentially from the input string.
 * It considers 'l' twice for "hello".
 *
 * @param str The input string to search within.
 * @return string: A string composed of characters from `str` that form "hello" as a subsequence.
 */
string FindHelloFromALine(string str) {
  char previous = ' ';    // Stores the previous character processed (not explicitly used for hello logic here).
  short int count_l = 0;  // Counts occurrences of 'l' for "hello" to handle both 'l's.
  string temp;            // Builds the "hello" subsequence found so far.

  /**
   * Block Logic: Iterates through each character of the input string to find "hello" as a subsequence.
   * Invariant: `temp` holds the longest prefix of "hello" found as a subsequence up to current `i`.
   */
  for (int i = 0; i < str.size(); i++) {
    // Conditional Logic: Checks if the current character is one of 'h', 'e', 'l', 'o'.
    if (str[i] == 'h' || str[i] == 'e' || str[i] == 'l' || str[i] == 'o') {
      if (str[i] == 'l')
        count_l++; // Increments 'l' count.

      // Conditional Logic: Determines if the character should be appended to `temp`.
      // It's appended if:
      // 1. The character is not already in `temp` (to maintain subsequence order and unique characters for h, e, o).
      // 2. OR, if the character is 'l' and we've encountered it for the second time (for the two 'l's in "hello").
      if ((!IfCharInString(str[i], temp)) || (str[i] == 'l' && count_l == 2)) {
        temp += str[i]; // Appends the character to the temporary subsequence.
      }
    }
    previous = str[i]; // Updates the previous character (unused in this specific logic).
  }

  return temp; // Returns the extracted subsequence.
}

int main() {
  string s; // Declares 's' to store the input string.

  // Reads the input string from standard input.
  cin >> s;

  // Calls the helper function to find the "hello" subsequence.
  string output = FindHelloFromALine(s);

  // Conditional Logic: Checks if the extracted subsequence has exactly 5 characters.
  // This implies that "hello" (h,e,l,l,o) was successfully formed.
  if (output.size() == 5)
    cout << "YES" << endl; // Prints "YES" if "hello" can be formed.
  else
    cout << "NO" << endl;  // Prints "NO" otherwise.

  return 0; // Indicates successful program execution.
}

/**
ahhellllloou
output:
YES

hlelo
output:
NO

pnnepelqomhhheollvlo
output:
YES
**/
