/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" competitive programming problem.
 * @details This program reads a single word and capitalizes its first letter if it is
 * currently in lowercase, leaving the rest of the word unchanged.
 *
 * @section algorithm Algorithm
 * The algorithm uses ASCII value manipulation to perform the case conversion:
 * 1. Read the input word into a string `a`.
 * 2. Check if the ASCII value of the first character is greater than 96. This is a
 *    method to determine if the character is a lowercase English letter (ASCII for 'a'
 *    is 97, 'z' is 122).
 * 3. If the character is indeed lowercase, subtract 32 from its ASCII value. This
 *    converts it to the corresponding uppercase character (e.g., 'a' (97) becomes 'A' (65)).
 * 4. Print the modified string.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the string, dominated by the I/O operation. The capitalization logic itself is O(1).
 * - Space Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string a; // Variable to store the input word.
  cin >> a;

  // Check if the string is not empty and its first character is a lowercase letter.
  // The ASCII value for 'a' is 97, so '> 96' is a simple check for lowercase.
  if (!a.empty() && (int)a[0] > 96) {
    // Convert the character to uppercase by subtracting 32 (the ASCII 'a' - 'A' distance).
    a[0] -= 32;
  }
  
  // Print the resulting string.
  cout << a;
}