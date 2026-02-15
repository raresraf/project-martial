/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" competitive programming problem.
 * @details The problem requires processing a string according to two rules:
 * 1. Delete all the vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Before each consonant, insert a single character ".".
 * All letters in the input string are converted to lowercase before processing.
 *
 * @section algorithm Algorithm
 * The algorithm iterates through the input string character by character and builds a new result string:
 * 1. Read the input string `inp`.
 * 2. Create an empty string `res` to store the result.
 * 3. Loop through each character of `inp`.
 * 4. In each iteration, convert the current character to its lowercase equivalent.
 * 5. Check if the lowercase character is a vowel ('a', 'e', 'i', 'o', 'u', 'y').
 * 6. If it is a vowel, the character is ignored, and the loop continues to the next character.
 * 7. If it is a consonant, a '.' character is appended to the `res` string, followed by
 *    the lowercase consonant itself.
 * 8. After the loop completes, print the final constructed string `res`.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the input string, as the program iterates through the string once.
 * - Space Complexity: O(L) to store the input and result strings. In the worst case (a string of all consonants), the result string will have a length of 2L.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string inp, res; // `inp` for input, `res` for the result string.
  cin >> inp;
  
  // Iterate through each character of the input string.
  for (int i = 0; i < inp.size(); i++) {
    // Convert the character to lowercase for case-insensitive processing.
    char c = tolower(inp[i]);
    
    // Check if the character is a vowel. In this problem, 'y' is also considered a vowel.
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y') {
      // If it's a vowel, do nothing and skip to the next character.
      continue;
    } else {
      // If it's a consonant, prepend a '.' and append the consonant to the result string.
      res += '.';
      res += c;
    }
  }
  
  // Print the final transformed string.
  cout << res << endl;
}
