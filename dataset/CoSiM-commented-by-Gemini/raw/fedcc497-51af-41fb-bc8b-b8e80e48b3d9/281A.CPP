/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem on Codeforces.
 *
 * @details The program takes a single word as input and capitalizes its first
 * letter if it is a lowercase letter. The rest of the word remains unchanged.
 *
 * Algorithm:
 * 1. Read the input word into a string.
 * 2. Check if the first character is a lowercase letter by checking if its
 *    ASCII value is within the range [97, 122] ('a' to 'z').
 * 3. If it is a lowercase letter, convert it to uppercase by subtracting 32
 *    from its ASCII value. This is the ASCII difference between a lowercase
 *    and an uppercase letter (e.g., 'a' - 'A' = 32).
 * 4. Print the modified string.
 *
 * Time Complexity: O(1), as only the first character is accessed and modified.
 * Space Complexity: O(L) to store the word of length L.
 */
#include "string"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  string s;
  cin >> s;

  // Check if the first character is a lowercase letter (ASCII 'a' to 'z').
  if (s[0] >= 97 && s[0] <= 122)
    // Convert to uppercase by subtracting the ASCII offset.
    s[0] -= 32;
  cout << s;

  return 0;
}
