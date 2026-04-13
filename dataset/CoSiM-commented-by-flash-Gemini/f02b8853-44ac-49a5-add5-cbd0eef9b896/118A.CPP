/* Problem Statements
*  A. String Task
time limit per test2 seconds
memory limit per test256 megabytes
inputstandard input
outputstandard output

Petya started to attend programming lessons. On the first lesson his task was to
write a simple program. The program was supposed to do the following: in the
given string, consisting if uppercase and lowercase Latin letters, it:

deletes all the vowels,
inserts a character "." before each consonant,
replaces all uppercase consonants with corresponding lowercase ones.

Vowels are letters "A", "O", "Y", "E", "U", "I", and the rest are consonants.
The program's input is exactly one string, it should return the output as a
single string, resulting after the program's processing the initial string.

Help Petya cope with this easy task.

Input
The first line represents input string of Petya's program. This string only
consists of uppercase and lowercase Latin letters and its length is from 1 to
100, inclusive.

Output
Print the resulting string. It is guaranteed that this string is not empty.

Examples
input
tour
output
.t.r
input
Codeforces
output
.c.d.f.r.c.s
input
aBAcAba
output
.b.c.b
*/

/*
 *  Author   : Byungwook Lee
 *  Date     : 2016/10/24
 *  E-mail   : quddnr153@gmail.com
 *  Github   : github.com/quddnr153
 *  Filename : 118A_stringtask.cpp
 */

/**
 * @file 118A.CPP
 * @brief Solution to a competitive programming problem: "A. String Task".
 *
 * This program transforms an input string by applying a set of rules:
 * 1. Deletes all vowels (A, O, Y, E, U, I, case-insensitive).
 * 2. Inserts a period ('.') character before each consonant.
 * 3. Replaces all uppercase consonants with their corresponding lowercase ones.
 *
 * Algorithm: Character-by-character processing with conditional logic to identify
 *            vowels/consonants and apply transformations.
 * Time Complexity: O(L), where L is the length of the input string. The algorithm
 *                  iterates through the string once.
 * Space Complexity: O(L) for the input string storage. The output string is generated
 *                   character by character, contributing effectively O(L) to overall space.
 */
#include <iostream> // Required for input/output operations (cin, cout).
#include <string.h> // Includes C-style string functions (though std::string is used).
#include <string>   // Required for std::string and its functionalities.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

/**
 * @brief Main function: entry point of the program.
 * Reads an input string and applies a series of transformations
 * (vowel deletion, consonant prefixing, uppercase to lowercase conversion)
 * before printing the modified string.
 */
int main() {
  string input; // Declares a string variable to store the input.
  cin >> input; // Reads the input string from standard input.

  /**
   * @brief Processes each character of the input string to apply transformation rules.
   * Block Logic: Iterates through the input string. For each character, it first
   * checks if it's a vowel (case-insensitive). If it is, the character is skipped.
   * If it's a consonant, a period is printed, and then the consonant itself is
   * printed, converted to lowercase if it was originally uppercase.
   * Pre-condition: `input` contains the string to be transformed.
   */
  for (int i = 0; i < input.size(); i++) {
    // Check for uppercase vowels and skip them.
    if (input[i] == 'A' || input[i] == 'O' || input[i] == 'Y' ||
        input[i] == 'E' || input[i] == 'U' || input[i] == 'I')
      continue; // Inline: Skips processing for uppercase vowels.
    // Check for lowercase vowels and skip them.
    if (input[i] == 'a' || input[i] == 'o' || input[i] == 'y' ||
        input[i] == 'e' || input[i] == 'u' || input[i] == 'i')
      continue; // Inline: Skips processing for lowercase vowels.

    // If the character is not a vowel, it must be a consonant.
    // Insert a period before the consonant.
    cout << '.';

    // Check if the consonant is uppercase. 'input[i] - 'A' >= 26' effectively
    // checks if `input[i]` is a lowercase letter by comparing its ASCII value
    // offset from 'A'. For example, 'a' - 'A' = 32, which is >= 26.
    if (input[i] - 'A' >= 26) // If true, the character is already lowercase.
      cout << input[i];       // Prints the lowercase consonant as is.
    else // The character is an uppercase consonant, convert it to lowercase.
      // Inline: Converts the uppercase consonant to lowercase using ASCII arithmetic.
      // 'a' - 'A' gives the difference in ASCII values between corresponding
      // lowercase and uppercase letters. Adding this difference to an uppercase
      // letter yields its lowercase equivalent.
      cout << (char)(input[i] + ('a' - 'A'));
  }
  cout << endl; // Prints a final newline character.
  return 0;     // Indicates successful program execution.
}