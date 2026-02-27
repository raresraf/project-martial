/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces using C-style I/O.
 * @details The program takes a single word as input and capitalizes its first
 * letter, leaving the rest of the word unchanged.
 *
 * @algorithm
 * 1. Read the input word into a C-style character array.
 * 2. Access the first character of the array (at index 0).
 * 3. Use the `toupper()` function to convert this character to its uppercase equivalent.
 * 4. Assign the result back to the first position in the array, modifying it in-place.
 * 5. Print the modified character array as a string.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the word, dominated by the I/O.
 * The capitalization itself is O(1).
 * Space Complexity: O(L_max), for the character array buffer, where L_max is
 * the maximum possible length (statically set to 1001 here).
 */
#include <cstdio>
#include <ctype.h>
#include <iostream>
#include <string>
using namespace std;

// The command-line arguments argc and argv are declared but not used.
int main(int argc, char *argv[]) {
  // A fixed-size character array to store the input word.
  // Note: Using scanf with %s can be unsafe and lead to buffer overflows.
  char word[1001];
  scanf("%s", word);
  
  // Capitalize the first letter of the word in-place.
  word[0] = toupper(word[0]);
  
  printf("%s\n", word);
  return 0;
}
