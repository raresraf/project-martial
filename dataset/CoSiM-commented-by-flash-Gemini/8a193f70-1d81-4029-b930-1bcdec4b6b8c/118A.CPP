/**
 * @8a193f70-1d81-4029-b930-1bcdec4b6b8c/118A.CPP
 * @brief This program processes an input string by converting uppercase consonants to lowercase,
 * effectively removing vowels (both upper and lower case), and then printing the remaining
 * consonant characters, each prefixed with a dot.
 * Algorithm: String manipulation, character-by-character processing.
 * Time Complexity: O(N), where N is the length of the input string, due to two linear passes.
 * Space Complexity: O(N) to store the input string.
 */

#include <cstring> // Required for strlen
#include <iostream>

using namespace std;

int main() {
  // @var i: Loop counter for iterating through the string.
  int i;
  // @var str: Character array to store the input string. Max size 100.
  char str[100];
  
  // Block Logic: Read the input string from standard input.
  cin >> str;
  
  // Block Logic: First pass - process the string to convert consonants and mark vowels.
  // Invariant: After each iteration, str[0...i] is processed.
  // Pre-condition: str contains the input string.
  for (i = 0; i < strlen(str); i++) {
    // Block Logic: Check if the current character is a vowel (case-insensitive).
    // If it's a vowel, replace it with a placeholder '-'.
    if ((str[i] == 'A') || (str[i] == 'O') || (str[i] == 'Y') ||
        (str[i] == 'E') || (str[i] == 'U') || (str[i] == 'I') ||
        (str[i] == 'a') || (str[i] == 'o') || (str[i] == 'y') ||
        (str[i] == 'e') || (str[i] == 'u') || (str[i] == 'i'))
      str[i] = '-';
    // Block Logic: Check if the current character is an uppercase consonant.
    // If it is, convert it to lowercase.
    else if ((64 < (int)str[i]) && ((int)str[i] < 91)) // ASCII check for uppercase letters (A-Z)
      str[i] = (char)((int)str[i] + 32); // Convert to lowercase (ASCII difference is 32)
  }
  
  // Block Logic: Second pass - print the processed string, skipping placeholders and adding dots.
  // Invariant: Characters in str[0...i] that are not '-' are printed with a prefix dot.
  // Pre-condition: str has been processed in the first pass.
  for (i = 0; i < strlen(str); i++) {
    // Block Logic: Only print characters that are not the vowel placeholder.
    if (str[i] != '-')
      cout << "." << str[i];
  }
  
  // Post-condition: A newline character is printed after the processed string.
  cout << endl;
  return 0; // Explicitly return 0 for successful execution.
}
