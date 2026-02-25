/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces (Problem 281A).
 *        This program reads a single word, capitalizes its first letter if it's
 *        originally lowercase, and then prints the modified word.
 * Algorithm: Character Case Conversion and String Concatenation
 * Time Complexity: O(L), where L is the length of the input string. The
 *                  character conversion is O(1), and printing the string is O(L).
 * Space Complexity: O(L) to store the input string.
 */

#include <bits/stdc++.h> // Includes various standard libraries, e.g., iostream, cstdio, string.
using namespace std;

/**
 * @brief Main function to capitalize the first letter of an input word.
 * @return 0 if the program executes successfully.
 */
int main() {
  // freopen("input.txt", "r", stdin); // Commented out: for redirecting input from a file.
  
  char ch;     // Declare a character variable to store the first character of the word.
  string s;    // Declare a string to store the rest of the word.
  
  // Read the first character using scanf.
  // Note: If previous input left a newline character in the buffer, this scanf
  // might consume it, leading to unexpected behavior for the subsequent cin >> s.
  scanf("%c", &ch);
  // Read the rest of the word using cin.
  cin >> s;
  
  /**
   * Block Logic: Checks if the first character 'ch' is a lowercase letter and capitalizes it if true.
   * Pre-condition: 'ch' contains the first character of the word.
   * Post-condition: 'ch' is printed as uppercase if it was lowercase, otherwise as is.
   */
  if (ch >= 'a' && ch <= 'z')
    // Inline: Convert lowercase 'ch' to uppercase by adjusting its ASCII value.
    printf("%c", ch - 'a' + 'A');
  else
    printf("%c", ch); // Print 'ch' as is if it's not a lowercase letter.
    
  // Print the rest of the word (string 's') followed by a newline.
  cout << s << endl;
  return 0; // Indicate successful program termination.
}
