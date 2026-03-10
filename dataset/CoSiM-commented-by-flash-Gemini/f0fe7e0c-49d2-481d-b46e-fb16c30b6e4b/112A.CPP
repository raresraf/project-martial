/**
 * @file 112A.CPP
 * @brief This program solves the "Petya and Strings" problem (Codeforces 112A).
 *   It compares two input strings lexicographically, ignoring case.
 *   Problem description: https://codeforces.com/problemset/problem/112/A
 * * Algorithm: Case-insensitive lexicographical comparison by converting to uppercase.
 *   1. Reads two input strings.
 *   2. Iterates through both strings, converting any lowercase characters to uppercase in-place.
 *   3. Compares the characters of the two strings (now effectively uppercase) one by one.
 *   4. The first differing characters determine the lexicographical order, and the loop breaks early.
 * Time Complexity: O(L) where L is the length of the strings (assuming they are of similar length),
 *   due to iterating through the strings once for case conversion and comparison.
 * Space Complexity: O(L) to store the input strings (modified in-place).
 */
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string.

using namespace std;

/**
 * @brief Main function to perform case-insensitive string comparison.
 * Functional Utility: Reads two strings, converts them to uppercase in-place,
 *                     compares them lexicographically character by character,
 *                     and prints the comparison result (-1, 0, or 1).
 */
int main() {
  string first;  // Stores the first input string.
  string second; // Stores the second input string.
  int diff = 0;  // Stores the comparison result: 0 for equal, 1 if first > second, -1 if first < second.
  // Block Logic: Reads the two input strings.
  cin >> first;
  cin >> second;
  /**
   * Block Logic: Iterates through the strings, converting characters to uppercase and performing comparison.
   * Pre-condition: `first` and `second` are populated with input strings.
   * Invariant: `first` and `second` are converted to uppercase in-place. `diff` is set upon the first difference.
   */
  for (int i = 0; i < first.length(); i++) {
    // Block Logic: Converts `first[i]` to uppercase if it's lowercase.
    char temp = first.at(i); // Temporary storage for current character.
    if (temp >= 'a' && temp <= 'z') { // Conditional Logic: Checks if character is lowercase using ASCII range.
      first[i] = temp - 32;          // Converts to uppercase by subtracting 32 from ASCII value.
    }
    // Block Logic: Converts `second[i]` to uppercase if it's lowercase.
    temp = second.at(i);
    if (temp >= 'a' && temp <= 'z') { // Conditional Logic: Checks if character is lowercase using ASCII range.
      second[i] = temp - 32;         // Converts to uppercase by subtracting 32 from ASCII value.
    }
    /**
     * Conditional Logic: Compares the characters at the current position.
     * If `first[i]` is greater than `second[i]`, `first` is lexicographically larger.
     */
    if (first[i] > second[i]) {
      diff = 1;  // Sets `diff` to 1.
      break;     // Breaks loop as order is determined.
    }
    /**
     * Conditional Logic: If `first[i]` is less than `second[i]`, `first` is lexicographically smaller.
     */
    else if (first[i] < second[i]) {
      diff = -1; // Sets `diff` to -1.
      break;     // Breaks loop as order is determined.
    }
  }
  // Block Logic: Prints the final comparison result (`diff`). If no difference found, `diff` remains 0.
  cout << diff;
  return 0;
}
