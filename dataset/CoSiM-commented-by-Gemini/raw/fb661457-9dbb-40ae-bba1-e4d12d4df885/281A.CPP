/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces (Problem 281A).
 * @see https://codeforces.com/problemset/problem/281/A
 *
 * @description
 * The program takes a single word as input and capitalizes its first letter.
 * If the first letter is already capitalized, it remains unchanged. The rest of
 * the word is not affected.
 *
 * @algorithm
 * 1. Read the input word into a string.
 * 2. Access the first character of the string (at index 0).
 * 3. Use the `toupper()` function to convert this character to its uppercase equivalent.
 * 4. Update the character at index 0 with the new capitalized letter.
 * 5. Print the modified string.
 *
 * @time_complexity O(L), where L is the length of the input word. This is
 *                  dominated by the string input/output operations.
 * @space_complexity O(L) to store the word in a string variable.
 */
#include <iostream>

using namespace std;

int main() {
  // Declare a string to hold the input word.
  string str;
  
  // Read the word from standard input.
  cin >> str;
  
  // Capitalize the first character of the string.
  // The toupper() function handles cases where the character is already uppercase
  // or not a letter.
  str[0] = toupper(str[0]);
  
  // Print the modified string to standard output.
  cout << str << endl;
  
  return 0;
}
