/**
 * @file 281A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 281A - "Word Capitalization").
 *
 * This program capitalizes the first letter of a given input word or phrase.
 * This implementation manually handles character case conversion using ASCII values.
 *
 * Algorithm: String manipulation with manual ASCII-based capitalization.
 * Reads an entire line of text, checks if the first character is a lowercase letter
 * by its ASCII value, converts it to uppercase by subtracting 32 (ASCII difference
 * between lowercase and uppercase letters), and then prints the entire modified string.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * This includes reading the string, calculating its length, and iterating through it once for printing.
 * Space Complexity: O(L) for storing the input string in a character array.
 */
#include <bits/stdc++.h> // Includes various standard libraries, including cstring for strlen.
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads a line of text, capitalizes its first letter using manual ASCII conversion, and prints the result.
 */
int main() {
  int d, c = 0, n; // 'd' for string length, 'c' is unused, 'n' for capitalized character ASCII.
  char ch[1000]; // Declare a character array to store the input line.
                 // Maximum length of 999 characters plus null terminator.
  // Pre-condition: Read an entire line of text (including spaces) into 'ch'.
  cin.getline(ch, 1000);
  // Functional Utility: Determine the length of the input string.
  d = strlen(ch);
  // Block Logic: Process only the first character for capitalization.
  // Invariant: This loop runs at most once to check and modify the first character.
  for (int i = 0; i < 1; i++) {
    // Pre-condition: Check if the first character is a lowercase letter (ASCII 'a' through 'z').
    if (ch[0] >= 97 && ch[0] <= 122) {
      // Functional Utility: Convert lowercase to uppercase by subtracting 32 from its ASCII value.
      n = ch[0] - 32;
      printf("%c", n); // Print the capitalized character.
      break;           // Exit the loop after processing the first character.
    } else {
      cout << ch[0]; // If not a lowercase letter, print it as is.
      break;         // Exit the loop.
    }
  }
  // Block Logic: Print the rest of the string starting from the second character.
  // Invariant: 'i' iterates from the second character to the end of the string.
  for (int i = 1; i < d; i++) {
    cout << ch[i];
  }
  cout << endl; // Add a newline character for cleaner output.
  return 0;
}
