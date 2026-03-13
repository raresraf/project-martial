/**
 * @file 112A.CPP
 * @brief Solution for CodeForces Problem 112A - "Petya and Strings".
 * 
 * @details
 * This program performs a case-insensitive lexicographical comparison of two input strings.
 * It manually converts both strings to uppercase, character by character, and then
 * compares them to find the lexicographical order.
 *
 * @algorithm Manual case-insensitive string comparison
 * 1. Read two strings, `a` and `b`.
 * 2. Loop through the strings character by character.
 * 3. In each iteration, convert the current characters of both strings to uppercase.
 *    - This is done by checking the ASCII value and subtracting 32 if it's a lowercase letter.
 * 4. Compare the uppercase characters:
 *    - If `a[i] < b[i]`, string `a` comes first. Set result to -1 and break.
 *    - If `a[i] > b[i]`, string `b` comes first. Set result to 1 and break.
 *    - If they are equal, continue to the next character, setting the result to 0.
 * 5. Print the final result.
 *
 * @time_complexity O(L), where L is the length of the strings.
 * @space_complexity O(L) for storing the two strings.
 * 
 * @note The initial value of `k=2` is arbitrary; it gets overwritten in the first loop iteration.
 * The ASCII manipulation (`a[i] - 32`) is not as portable as using `toupper()`.
 */
#include <cstring>
#include <iostream>
#include <cctype> // For toupper

using namespace std;

int main() {
  string a, b;
  cin >> a >> b;

  int k = 0; // Stores the comparison result: -1, 0, or 1. Initialized to 0 for the case of equal strings.

  /**
   * @brief Main loop to perform character-by-character case-insensitive comparison.
   */
  for (int i = 0; i < a.length(); i++) {
    // Convert characters to uppercase for comparison.
    char char_a = toupper(a[i]);
    char char_b = toupper(b[i]);

    // Perform comparison.
    if (char_a < char_b) {
      k = -1;
      break; // First string is smaller, no need to check further.
    } else if (char_a > char_b) {
      k = 1;
      break; // Second string is smaller, no need to check further.
    }
  }

  cout << k << endl;

  return 0;
}
