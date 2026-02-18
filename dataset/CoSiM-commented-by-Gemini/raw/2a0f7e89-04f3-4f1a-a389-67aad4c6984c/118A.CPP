/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem (118A) from Codeforces.
 * @details This program processes a string according to three rules:
 * 1. Delete all vowels (A, O, Y, E, U, I).
 * 2. Insert a '.' before each consonant.
 * 3. Convert all uppercase consonants to lowercase.
 * This implementation performs all three steps in a single pass.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution iterates through the input string once, character by character.
 * 1. Read the input string.
 * 2. Loop through each character of the string.
 * 3. In each iteration, convert the current character to its lowercase equivalent.
 * 4. Check if the lowercase character is a vowel (a, e, i, o, u, y).
 * 5. If it is *not* a vowel (i.e., it's a consonant), print a '.' character followed
 *    by the lowercase consonant.
 * 6. After the loop, print a final newline.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the string, as it only requires a
 * single pass through the input.
 * Space Complexity: O(N) to store the input string.
 */
#include <cctype>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string in;
  cin >> in;
  /**
   * @block
   * @brief Iterates through the input string, processing one character at a time.
   * @pre `in` holds the original string.
   * @post The transformed string is printed to standard output.
   */
  for (int i = 0; i < in.size(); i++) {
    // Convert character to lowercase for case-insensitive vowel check.
    char c = tolower(in[i]);

    // Check if the character is a consonant.
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y')

    {
      // If it's a consonant, print a '.' followed by the lowercase character.
      cout << "." << c;
    }
    // cout<<in<<endl;
  }
  cout << endl;
  return 0;
}