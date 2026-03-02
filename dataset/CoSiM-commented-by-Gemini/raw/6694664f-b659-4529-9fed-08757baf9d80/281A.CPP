#include <bits/stdc++.h>
using namespace std;

/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem from Codeforces (Problem 281A).
 * @details This program reads a word and capitalizes its first letter.
 *
 * Algorithm:
 * The program reads a single word from standard input. It then isolates the first
 * character, converts it to uppercase using the `toupper()` function, and prints it.
 * Following this, it iterates through the rest of the characters of the word,
 * from the second character to the end, and prints them without any change.
 *
 * Time Complexity: O(L), where L is the length of the input string. This is because
 * the program iterates through the string once to print it.
 * Space Complexity: O(L), to store the input string.
 */
int main() {
  string str;
  char ch;
  int n, i;

  // Read the input word.
  cin >> str;
  n = str.size();

  // Capitalize the first character of the string.
  ch = toupper(str[0]);

  // Print the capitalized first character.
  cout << ch;

  /**
   * @brief Print the rest of the string.
   * This loop iterates from the second character to the end of the string
   * and prints each character.
   */
  for (i = 1; i < n; i++) {
    cout << str[i];
  }
  return 0;
}
