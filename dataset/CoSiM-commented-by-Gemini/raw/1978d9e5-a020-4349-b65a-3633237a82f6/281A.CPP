/**
 * @file 281A.CPP
 * @brief Solution for capitalizing the first letter of each word in a line.
 * @details This program reads a line of text and converts the first character of every
 * word to uppercase, effectively applying a "title case" format to the input string.
 *
 * @section algorithm Algorithm
 * 1. A C-style character array `str` is declared to hold the input line.
 * 2. `cin.getline()` is used to read an entire line of text, including spaces, into `str`.
 * 3. The length of the string is calculated using `strlen`.
 * 4. The very first character of the string, `str[0]`, is capitalized using `toupper`.
 * 5. The program then iterates through the entire string.
 *    a. Inside the loop, it checks if the current character `str[i]` is a space.
 *    b. If a space is found, it capitalizes the character immediately following the
 *       space, `str[i + 1]`.
 * 6. After the loop, the modified string is printed to the console using `cout.write`.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the input string. The program
 *   iterates through the string once to perform the capitalization.
 * - Space Complexity: O(L) to store the character array for the input string.
 */
#include <cstring>
#include <iostream>
#include <cctype> // For toupper

using namespace std;

/**
 * @brief Main function to apply title case to an input string.
 * @return 0 on successful execution.
 */
int main() {
  char str[1001]; // Character buffer to store the input line.
  
  // Read an entire line of input, up to 1000 characters.
  cin.getline(str, 1001);
  
  int n = strlen(str); // Get the length of the input string.
  
  // Capitalize the very first letter of the string.
  if (n > 0) {
    str[0] = toupper(str[0]);
  }
  
  // Loop through the string to find spaces.
  for (int i = 0; i < n - 1; i++) { // Loop until n-1 to safely access i+1
    // If a space is found, capitalize the next character.
    if (str[i] == ' ') {
      str[i + 1] = toupper(str[i + 1]);
    }
  }
  
  // Write the modified string to standard output.
  cout.write(str, n);
  
  return 0;
}
