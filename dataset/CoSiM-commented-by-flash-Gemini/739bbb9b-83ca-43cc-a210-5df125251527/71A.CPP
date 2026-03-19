/**
 * @file 71A.CPP
 * @brief Solution to Codeforces Problem 71A: "Way Too Long Words".
 *
 * This problem asks to abbreviate words that are longer than 10 characters.
 * The abbreviation rule is: the first character, followed by the count of characters
 * between the first and last (as a digit string), followed by the last character.
 * Words 10 characters or shorter remain unchanged. For example, "localization" becomes "l10n".
 *
 * Algorithm: For each input word, check its length. If it's greater than 10, apply
 * the abbreviation rule; otherwise, return the original word.
 *
 * Time Complexity: O(N * L_max), where N is the number of words and L_max is the
 *                  maximum length of a word. Each word processing takes O(L) time.
 * Space Complexity: O(L_max) to store the input word, its character vector, and the result string.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++, including iostream, string, and vector.
using namespace std;

/** @def ll
 * @brief Type alias for `long long`.
 */
#define ll long long
/** @def pb
 * @brief Alias for `push_back`, commonly used with `std::vector`.
 */
#define pb push_back
/** @def vc
 * @brief Type alias for `std::vector<char>`.
 */
#define vc vector<char>

/**
 * @brief Abbreviates a word if its length is greater than 10 characters.
 *
 * The abbreviation rule is: first character, followed by the count of characters
 * between the first and last (as a digit string), followed by the last character.
 * Words 10 characters or shorter are returned unchanged.
 *
 * @param s The input string (word) to potentially abbreviate.
 * @return The abbreviated or original string.
 */
string newword(string s) {
  // Converts the input `std::string` to a `std::vector<char>`.
  // This conversion is not strictly necessary as `std::string` provides direct
  // character access and length, but is used in this specific solution.
  vc a(s.begin(), s.end());

  int p = a.size(); // Stores the length of the word.
  string ans = "";  // Initializes an empty string to build the result.

  /**
   * Block Logic: Checks if the word's length exceeds 10 characters to determine abbreviation.
   * Pre-condition: `s` is the input word, `p` is its length.
   * Invariant: `ans` will hold either the abbreviated or original word.
   */
  if (p > 10) {
    // long word
    char start = a[0];             // Get the first character.
    char end = a[a.size() - 1];    // Get the last character.
    int count = 0;                 // Counter for characters between first and last.
    /**
     * Block Logic: Calculates the count of characters between the first and last characters.
     * This loop is functionally equivalent to `count = a.size() - 2`, but is implemented
     * as an explicit iteration here.
     */
    for (size_t i = 1; i < a.size() - 1; i++) {
      count += 1; // Increment count for each middle character.
    }
    // Construct the abbreviated string: first char + count_as_string + last char.
    ans += start + to_string(count) + end;
  } else {
    // return the string as it is
    ans += s; // If 10 characters or shorter, use the original word.
  }
  return ans; // Return the resulting string.
}

/**
 * @brief Main function to solve the "Way Too Long Words" problem.
 * Reads the number of words, then processes each word for abbreviation.
 */
int main() {
  int n; // Stores the number of words (test cases) to process.
  cin >> n; // Reads the number of words.

  /**
   * Block Logic: Processes each word one by one.
   * Pre-condition: `n` specifies the number of words.
   * Invariant: After each iteration, one word has been read, processed, and its result printed.
   */
  while (n--) {
    string s; // Stores the current word read from input.
    cin >> s; // Reads the current word.
    cout << newword(s) << "\n"; // Calls `newword` to get the result and prints it.
  }
  return 0; // Indicates successful execution.
}
