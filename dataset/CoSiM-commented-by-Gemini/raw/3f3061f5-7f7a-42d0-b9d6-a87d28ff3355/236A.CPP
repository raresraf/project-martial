/**
 * @file 236A.CPP
 * @brief An inefficient solution to the "Boy or Girl" problem (Codeforces 236A).
 *
 * @algorithm
 * The problem is to determine if a username has an even or odd number of
 * distinct characters.
 *
 * This solution uses a brute-force, alphabet-scanning approach:
 * 1. Read the input username string.
 * 2. Initialize a counter for distinct characters to zero.
 * 3. Loop through every character of the English alphabet (from 'a' to 'z').
 * 4. For each alphabet character, perform a linear scan through the entire
 *    input string to see if the character exists.
 * 5. If it exists, increment the distinct character counter.
 * 6. After checking all 26 letters, check the parity (even or odd) of the
 *    counter and print the corresponding message.
 *
 * @complexity
 * - Time Complexity: O(K * L), where K is the size of the alphabet (26) and L
 *   is the length of the string. This is inefficient compared to O(L) or
 *   O(L log L) solutions using sets, maps, or sorting.
 * - Space Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string> // Recommended for std::string
using namespace std;
int main() {
  string s;
  int flag;       // Flag to indicate if a character from the alphabet is found.
  int c = 0;      // Counter for the number of distinct characters found.
  cin >> s;

  /**
   * @brief Outer loop iterates through each character of the lowercase alphabet.
   */
  for (int i = 'a'; i <= 'z'; i++) {
    flag = 0; // Reset flag for each new character to check.
    /**
     * @brief Inner loop scans the entire input string to find the current
     * alphabet character `i`.
     */
    for (int j = 0; j < s.length(); j++)
      if (s[j] == i)
        flag = 1; // Set flag if found.
    // If the character was found in the string, increment the distinct count.
    if (flag == 1)
      c++;
  }
  /**
   * @brief Check the parity of the distinct character count to determine output.
   */
  if (c % 2) // A non-zero result means `c` is odd.
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
}
