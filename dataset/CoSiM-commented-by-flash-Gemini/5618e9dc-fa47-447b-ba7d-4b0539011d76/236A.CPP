/**
 * @5618e9dc-fa47-447b-ba7d-4b0539011d76/236A.CPP
 * @brief This program determines an output phrase based on the parity (even or odd)
 * of the number of unique lowercase English characters in an input string.
 *
 * Algorithm: Character frequency counting to determine uniqueness, followed by parity check.
 * Time Complexity: O(L + C) where L is the length of the input string and C is the size of the character set (26).
 * Space Complexity: O(C) for the frequency array, effectively O(1).
 */

#include <cstring> // For memset, though not explicitly used in the provided snippet, often included with cstring operations.
#include <iostream> // For standard input/output operations (cin, cout).
#include <string> // For using std::string.

using namespace std; // Uses the standard namespace to avoid prefixing std::.

int main() {
  string s; // Declares a string variable to store the input username.
  cin >> s; // Reads the username string from standard input.
  
  // Declares an integer array of size 27 (0-26) initialized to zeros.
  // This array acts as a frequency map or a presence tracker for lowercase English letters.
  // Index 0-25 correspond to 'a'-'z'.
  int arr[27] = {0}; 
  int count = 0; // Initializes a counter for unique characters found in the string.
  
  /**
   * Block Logic: Iterates through each character of the input string to mark the presence
   * of unique characters in the 'arr' array.
   * Invariant: After each iteration, 'arr' accurately reflects whether characters
   * encountered so far are present in the string.
   */
  for (int i = 0; i < s.length(); i++) {
    // Inline: Marks the character at s[i] as present by setting the corresponding
    // index in 'arr' to 1. 's[i] - 97' converts the character's ASCII value
    // to a 0-25 index ('a' is 97 in ASCII).
    arr[s[i] - 97] = 1;
  }
  
  /**
   * Block Logic: Iterates through the 'arr' array to count the total number of
   * unique characters that were present in the input string.
   * Invariant: 'count' accumulates the total number of unique characters.
   */
  for (int i = 0; i < 26; i++) {
    // Inline: If 'arr[i]' is 1, it means the character corresponding to index 'i'
    // was found in the input string, so increment the unique character counter.
    if (arr[i] == 1)
      count++;
  }
  
  /**
   * Block Logic: Determines the output phrase based on the parity of the unique character count.
   * Pre-condition: 'count' holds the final tally of unique characters.
   * Invariant: One of two specific phrases is printed based on whether 'count' is even or odd.
   */
  if (count % 2 == 0)
    cout << "CHAT WITH HER!"; // Output if the number of unique characters is even.
  else
    cout << "IGNORE HIM!"; // Output if the number of unique characters is odd.
  
  return 0; // Indicates successful program termination.
}
