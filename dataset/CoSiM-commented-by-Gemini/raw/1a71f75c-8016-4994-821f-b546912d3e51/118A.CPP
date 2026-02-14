/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem.
 * @details This program processes a given string according to specific rules:
 * 1. It converts all uppercase letters to lowercase.
 * 2. It deletes all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 3. It inserts a character "." before each consonant.
 *
 * @section algorithm Algorithm
 * 1. Read the input string `inp`.
 * 2. Initialize an empty string `res` which will be used to build the result.
 * 3. Iterate through each character of the input string `inp`.
 * 4. In each iteration:
 *    a. Convert the current character to its lowercase equivalent.
 *    b. Check if the lowercase character is a vowel ('a', 'e', 'i', 'o', 'u', 'y').
 *    c. If it is a vowel, the loop continues to the next character, effectively deleting it.
 *    d. If it is a consonant, a '.' character is first appended to the `res` string,
 *       followed by the consonant itself.
 * 5. After the loop completes, the final transformed string `res` is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the input string. The program
 *   iterates through the string once.
 * - Space Complexity: O(L), to store the input and result strings. In the worst
 *   case (a string of all consonants), the result string will be twice the length
 *   of the input.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to execute the string transformation task.
 * @return 0 on successful execution.
 */
int main() {
  string inp, res;
  cin >> inp;
  
  // Iterate through each character of the input string.
  for (int i = 0; i < inp.size(); i++) {
    // Convert the character to lowercase for consistent checking.
    char c = tolower(inp[i]);
    
    // Check if the character is a vowel.
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y') {
      // If it's a vowel, skip it (effectively deleting it).
      continue;
    } else {
      // If it's a consonant, prepend a '.' and add it to the result.
      res += '.';
      res += c;
    }
  }
  
  cout << res << endl;
  return 0;
}
