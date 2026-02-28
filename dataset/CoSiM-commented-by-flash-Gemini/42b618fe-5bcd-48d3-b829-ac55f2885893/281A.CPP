/**
 * @file 281A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 281A - "Word Capitalization").
 *
 * This program capitalizes the first letter of a given input word.
 *
 * Algorithm: String manipulation.
 * Reads a single word as a string, converts its first character to uppercase,
 * and then prints the modified string.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * This includes reading the string, modifying the first character, and printing the string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h> // Includes various standard libraries, including cctype for toupper and cstring for strlen.
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads a word, capitalizes its first letter, and prints the result.
 */
int main() {
  char s[1003]; // Declare a character array to store the input word.
                // Maximum length of 1002 characters plus null terminator.
  // Pre-condition: Read the input word into character array 's'.
  cin >> s;
  // Functional Utility: Convert the first character of the string to uppercase.
  s[0] = toupper(s[0]);
  // Block Logic: Iterate through the characters of the modified string and print each one.
  // Invariant: 'i' traverses the string from the first character up to the null terminator.
  for (int i = 0; i < strlen(s); i++)
    cout << s[i];
  cout << endl; // Add a newline character for cleaner output.
  return 0;
}
