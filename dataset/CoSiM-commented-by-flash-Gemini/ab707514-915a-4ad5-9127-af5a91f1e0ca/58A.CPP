/**
 * @file 58A.CPP
 * @brief This program checks if the word "hello" can be formed as a subsequence from an input string.
 * It iterates through the input string, matching characters with "hello" in sequence.
 * Algorithm: Subsequence matching with a linear scan.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string, or O(1) if input is streamed.
 */

#include <iostream>

using namespace std;

/**
 * @brief Main function to determine if "hello" is a subsequence of the input string.
 * It reads an input string, attempts to find "hello" as a subsequence, and prints "YES" or "NO".
 * @return 0 on successful execution.
 */
int main() {
  // @var hello: String literal representing the target subsequence "hello".
  string hello = "hello";
  // @var a: Index for iterating through the 'hello' string, tracking the current character to match.
  // @var c: Counter for matched characters, indicating how many characters of "hello" have been found in sequence.
  int a = 0, c = 0;
  // @var palabra: String to store the input word.
  string palabra;
  // Read the input word from standard input.
  cin >> palabra;
  /**
   * @brief Loop through each character of the input string (`palabra`).
   * Invariant: At the start of each iteration, 'c' reflects the count of "hello" characters
   * found as a subsequence up to the current position in `palabra`.
   */
  for (int i = 0; i < palabra.size(); i++) {
    // Check if the current character in 'palabra' matches the character at index 'a' in "hello".
    if (palabra[i] == hello[a]) {
      // If a match is found, advance the index for "hello".
      a++;
      // Increment the count of matched characters.
      c++;
    }
  }
  // Determine if all 5 characters of "hello" were found as a subsequence.
  // If 'c' is 5, print "YES"; otherwise, print "NO".
  (c == 5) ? cout << "YES" : cout << "NO";
  return 0;
}
