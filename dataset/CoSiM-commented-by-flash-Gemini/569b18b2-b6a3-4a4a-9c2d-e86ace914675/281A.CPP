/**
 * @569b18b2-b6a3-4a4a-9c2d-e86ace914675/281A.CPP
 * @brief This program takes a single word as input and capitalizes its first letter.
 * It then prints the modified word to standard output.
 *
 * Algorithm: String manipulation - specifically, in-place modification of the first character.
 * Time Complexity: O(L) where L is the length of the input word, effectively O(1) due to fixed buffer size.
 * Space Complexity: O(L) for storing the word, effectively O(1) due to fixed buffer size.
 */

#include <cstdio>  // For standard input/output functions like scanf and printf.
#include <ctype.h> // For character handling functions like toupper.
#include <iostream> // Included but not strictly necessary as cstdio is used for I/O.
#include <string>  // Included but not strictly necessary as C-style char array is used for string.

using namespace std; // Uses the standard namespace, though not critical with direct C functions.

int main(int argc, char *argv[]) {
  // Declares a character array (C-style string) to store the input word.
  // The size 1001 allows for words up to 1000 characters plus a null terminator.
  char word[1001]; 
  
  // Block Logic: Reads a single word from standard input.
  // Pre-condition: Input is a single, space-delimited word.
  // Invariant: 'word' array contains the input string followed by a null terminator.
  scanf("%s", word); // Reads characters into 'word' until a whitespace character is encountered.
  
  // Block Logic: Converts the first character of the word to uppercase.
  // Pre-condition: 'word' contains at least one character.
  // Invariant: The first character of 'word' is its uppercase equivalent.
  // Inline: 'toupper' converts a character to its uppercase form if it's a lowercase letter; otherwise, it returns the character unchanged.
  word[0] = toupper(word[0]); 
  
  // Block Logic: Prints the modified word to standard output, followed by a newline character.
  // Invariant: The capitalized word is displayed to the user.
  printf("%s\n", word); 
  
  return 0; // Indicates successful program termination.
}
