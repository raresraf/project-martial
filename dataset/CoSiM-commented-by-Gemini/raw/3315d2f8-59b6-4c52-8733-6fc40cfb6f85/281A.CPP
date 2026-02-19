/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem.
 * @details This program capitalizes the first letter of an input word. It achieves this
 * by printing a modified first character and then printing the rest of the string separately.
 *
 * @algorithm The algorithm processes the word in two parts:
 * 1. The first character is read, converted to its uppercase equivalent, and printed.
 * 2. The program then iterates through the rest of the string from the second character
 *    to the end, printing each character as is.
 * This reconstructs the word with a capitalized first letter on the output stream
 * without modifying the original string in memory.
 *
 * @time_complexity O(L), where L is the length of the word, as the process involves
 * reading the string and then iterating through it once to print.
 * @space_complexity O(L), to store the input word in a string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  // Isolate the first character and convert it to uppercase.
  char x = toupper(s.at(0));
  // Print the capitalized first character.
  cout << x;
  /**
   * Block Logic: Print the remainder of the string from the second character onwards.
   */
  for (int i = 1; i < s.size(); i++)
    cout << s.at(i);
  return 0;
}
