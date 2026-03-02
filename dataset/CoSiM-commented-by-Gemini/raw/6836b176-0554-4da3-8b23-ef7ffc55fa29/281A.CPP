#include <cctype>
#include <cstdio>

using namespace std;

/**
 * @file 281A.CPP
 * @brief C-style solution to the "Word Capitalization" problem from Codeforces (Problem 281A).
 * @details This program reads a word and capitalizes its first letter using C-style
 * character arrays and I/O functions.
 *
 * Algorithm:
 * A character array is declared to hold the input word. The word is read from
 * standard input using `scanf`. The program then directly accesses the first element
 * of the array (the first character) and converts it to uppercase in-place using
 * `toupper`. Finally, the entire modified character array is printed to standard
 * output as a string using `printf`.
 *
 * Time Complexity: O(L), where L is the length of the input word, dominated by the
 * I/O operations.
 * Space Complexity: O(L_max), where L_max is the maximum possible length of the word,
 * due to the static allocation of the character array.
 */
int main() {
  char x[1001];
  // Read the input word using C-style formatted input.
  scanf("%s", &x);
  // Capitalize the first character of the string in-place.
  x[0] = toupper(x[0]);
  // Print the modified string using C-style formatted output.
  printf("%s", x);
  return 0;
}