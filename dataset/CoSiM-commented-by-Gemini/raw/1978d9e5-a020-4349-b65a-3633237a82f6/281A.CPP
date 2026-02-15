/**
 * @file 281A.CPP
 * @brief Solution for capitalizing the first letter of each word in a line of text.
 * @details This program reads a line of text and capitalizes the first letter of
 * the very first word and the first letter of any word that follows a space.
 * This correctly solves the "Word Capitalization" problem (281A) as a special case,
 * since its input is a single word.
 *
 * @section algorithm Algorithm
 * The program uses C-style strings and character manipulation:
 * 1. A character array `str` is declared to hold the input line.
 * 2. `cin.getline()` reads the entire line, preserving spaces.
 * 3. The first character of the entire string (`str[0]`) is capitalized using `toupper()`.
 * 4. A loop iterates through the string to find space characters.
 * 5. Whenever a space is found at index `i`, the character immediately following it
 *    (`str[i+1]`) is also capitalized, handling the start of subsequent words.
 * 6. Finally, the modified C-style string is printed to the console using `cout.write()`.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the input line, as the program iterates through the string once.
 * - Space Complexity: O(L) to store the character array for the input line.
 */
#include <cstring>  // For strlen
#include <iostream>
#include <cctype>   // For toupper

using namespace std;
int main() {
  // Declare a C-style character array to store the input string.
  char str[1000];
  
  // Read an entire line from standard input, including any spaces.
  cin.getline(str, 1000);
  
  // Get the length of the C-style string.
  int n = strlen(str);
  
  // Capitalize the first letter of the entire string, if the string is not empty.
  if (n > 0) {
    str[0] = toupper(str[0]);
  }
  
  // Loop through the string to capitalize the first letter of subsequent words.
  for (int i = 0; str[i] != '\0'; i++) {
    // Check for a space character.
    if (str[i] == ' ') {
      // If a space is found, capitalize the character that follows it,
      // ensuring it is not the null terminator.
      if (str[i + 1] != '\0') {
        str[i + 1] = toupper(str[i + 1]);
      }
    }
  }
  
  // Write the modified string content to standard output.
  cout.write(str, n);
  
  return 0;
}
