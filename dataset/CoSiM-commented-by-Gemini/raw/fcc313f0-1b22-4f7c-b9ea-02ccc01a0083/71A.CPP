/**
 * @file 71A.cpp
 * @brief Solution to the "Way Too Long Words" problem using C-style strings.
 * This program reads a number of words and, for each word, if its length is
 * greater than 10, it prints an abbreviation of it. Otherwise, it prints the
 * word as is. The abbreviation consists of the first character, the number of
 * characters between the first and last characters, and the last character.
 * This implementation uses C-style I/O and character arrays.
 *
 * Algorithm:
 * 1. Read the number of words `n`.
 * 2. Loop `n` times:
 *    a. Read a word into a character buffer.
 *    b. Calculate the length of the word using `strlen`.
 *    c. If the word's length is greater than 10, print the abbreviation.
 *    d. Otherwise, print the word.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the length of the longest word.
 * Space Complexity: O(L) for the character buffer.
 */
#include <cstdio>
#include <cstring>
using namespace std;

int main() {
  int n;
  char buffer[101]; // Buffer to hold the input word.
  int len;
  
  // Read the number of words to process.
  scanf("%d", &n);
  
  // Loop n times to process each word.
  while (n--) {
    scanf("%s", buffer);
    len = strlen(buffer);
    
    // If the word's length is greater than 10, abbreviate it.
    if (len > 10) {
      printf("%c", buffer[0]);
      printf("%d", len - 2);
      printf("%c", buffer[len - 1]);
    } else {
      // Otherwise, print the word as is.
      printf("%s", buffer);
    }
    printf("\n");
  }
}
