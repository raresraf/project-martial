/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (118A).
 * @details The problem specifies a three-step transformation for a given string:
 * 1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Insert a '.' character before each consonant.
 * 3. Replace all uppercase consonants with their lowercase equivalents.
 *
 * @algorithm
 * The solution implements the transformation in three main passes:
 * 1. **Filter Consonants:** It iterates through the input string and copies only the
 *    consonants into a new character array `x`. Vowels (both upper and lower case)
 *    are ignored.
 * 2. **Convert to Lowercase:** It iterates through the new array `x` (containing only
 *    consonants) and converts any uppercase characters to lowercase.
 * 3. **Format Output:** It iterates through the final, all-lowercase consonant array `x`
 *    and prints each character prefixed with a '.'.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the input string, as the solution
 * involves several linear passes over the string or derived data.
 * Space Complexity: O(N), as it uses auxiliary character arrays to store the
 * intermediate and final results.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @variable arr: Stores the initial input string.
   * @variable x: Stores the filtered consonants from the input string.
   */
  char arr[100], x[200];
  cin >> arr;
  
  int l = strlen(arr); // Length of the input string.
  int c = 0;           // Counter for the number of consonants found.
  int i;               // Loop counter.
  
  // This line is redundant as `cin >> arr` ensures null-termination.
  arr[l] = '\0';
  
  /**
   * @block
   * @brief First pass: Filter out vowels and copy consonants to array `x`.
   * Vowels are defined as 'A', 'E', 'I', 'O', 'U', 'Y' and their lowercase versions.
   */
  for (i = 0; i < l; i++) {
    // Pre-condition: Check if the character is NOT a vowel.
    if (arr[i] != 'A' && arr[i] != 'a' && arr[i] != 'E' && arr[i] != 'e' &&
        arr[i] != 'I' && arr[i] != 'i' && arr[i] != 'O' && arr[i] != 'o' &&
        arr[i] != 'U' && arr[i] != 'u' && arr[i] != 'Y' && arr[i] != 'y') {
      // If it's a consonant, add it to the `x` array.
      x[c] = arr[i];
      c++;
    }
  }

  /**
   * @block
   * @brief Second pass: Convert all characters in the consonant array `x` to lowercase.
   */
  for (i = 0; i < c; i++) {
    if (isupper(x[i]))
      x[i] = tolower(x[i]);
  }
  
  /**
   * @block
   * @brief Third pass: Print the formatted output.
   * Each consonant is prefixed with a '.'.
   */
  for (i = 0; i < c; i++)
    cout << "." << x[i];
    
  cout << endl;
  return 0;
}