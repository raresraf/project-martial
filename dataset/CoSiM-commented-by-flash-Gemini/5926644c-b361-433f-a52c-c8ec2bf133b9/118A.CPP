/**
 * @5926644c-b361-433f-a52c-c8ec2bf133b9/118A.CPP
 * @brief This program transforms an input string by first removing all vowels (case-insensitively),
 * then converting the remaining characters (consonants) to lowercase, and finally printing
 * each consonant prefixed with a dot ('.').
 *
 * Algorithm: Multi-pass string processing involving filtering, case conversion, and formatted output.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(L) for storing the input and intermediate consonant arrays.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes many standard libraries like cstring, iostream, cctype.
using namespace std;     // Uses the standard namespace to avoid prefixing std::.

int main() {
  char arr[100], x[200]; // Declares 'arr' for input (max 99 chars) and 'x' for filtered consonants (max 199 chars).
  cin >> arr;           // Reads the input string into 'arr'.
  int l = strlen(arr), c = 0, i; // 'l' stores length of 'arr', 'c' counts consonants, 'i' is loop counter.
  arr[l] = '\0'; // Ensures null termination for 'arr' if `cin >> arr` doesn't guarantee it for full buffer. (Though `cin >> char_array` usually handles this).
  
  /**
   * Block Logic: Filters out vowels from the input string 'arr' and copies consonants to array 'x'.
   * Invariant: 'x' contains only the consonants from 'arr', in their original case, and 'c' is the count of these consonants.
   */
  for (i = 0; i < l; i++) { // Iterates through each character of the input string.
    // Inline: Checks if the current character is NOT any of the vowels (case-insensitively).
    if (arr[i] != 'A' && arr[i] != 'a' && arr[i] != 'E' && arr[i] != 'e' &&
        arr[i] != 'I' && arr[i] != 'i' && arr[i] != 'O' && arr[i] != 'o' &&
        arr[i] != 'U' && arr[i] != 'u' && arr[i] != 'Y' && arr[i] != 'y') {
      x[c] = arr[i]; // Copies the consonant to 'x'.
      c++;           // Increments the consonant counter.
    }
  }
  
  /**
   * Block Logic: Converts all consonants stored in array 'x' to lowercase.
   * Invariant: All characters in 'x' are lowercase after this loop.
   */
  for (i = 0; i < c; i++) { // Iterates through the array of consonants.
    // Inline: Checks if the character is uppercase and converts it to lowercase if it is.
    if (isupper(x[i])) 
      x[i] = tolower(x[i]);
  }
  
  /**
   * Block Logic: Prints each processed consonant from 'x', prefixed with a dot.
   * Invariant: The transformed string is output to standard output.
   */
  for (i = 0; i < c; i++)
    cout << "." << x[i]; // Prints a dot followed by the consonant.
  cout << endl; // Prints a newline character at the end of the output.
  return 0;     // Indicates successful program termination.
}