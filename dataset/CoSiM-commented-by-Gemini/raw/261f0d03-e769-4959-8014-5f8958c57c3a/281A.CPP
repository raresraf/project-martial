/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem (281A) from Codeforces.
 * @details This program capitalizes the first letter of a given word.
 *
 * @algorithm The program reads a word and directly manipulates the first character.
 * 1. Read the input word into a string.
 * 2. Check if the first character is a lowercase English letter.
 * 3. If it is, convert it to its uppercase equivalent.
 * 4. Print the modified string.
 * The check `s[0] > 'Z'` is a simplified way to check for lowercase, assuming an ASCII-compatible character set where lowercase letters follow uppercase letters.
 * The conversion `s[0] -= 32` relies on the ASCII encoding where the difference between a lowercase and uppercase letter is 32.
 *
 * @complexity
 * Time Complexity: O(L) - where L is the length of the string, dominated by I/O operations. The capitalization logic itself is O(1).
 * Space Complexity: O(L) - to store the input string.
 */
#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  string s;
  cin >> s;
  /**
   * @block
   * @brief Capitalizes the first letter of the string if it is lowercase.
   * @pre 's' is a non-empty string.
   * @post The first character of 's' is an uppercase letter if it was originally a lowercase English letter.
   */
  if (s[0] > 'Z') // A simple check for lowercase letters in ASCII.
    s[0] -= 32;   // Converts lowercase to uppercase in ASCII by subtracting 32.
  cout << s << "\n";
}
