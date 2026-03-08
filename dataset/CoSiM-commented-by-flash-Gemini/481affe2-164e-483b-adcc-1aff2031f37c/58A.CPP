/**
 * @481affe2-164e-483b-adcc-1aff2031f37c/58A.CPP
 * @brief This program determines if the string "hello" is a subsequence of a given input string.
 * Algorithm: Subsequence check using two pointers.
 * Time Complexity: O(N) where N is the length of the input string, as it involves a single pass.
 * Space Complexity: O(1) beyond input storage, as only a few variables are used.
 */
#include <iostream>
using namespace std;

string a, b = "hello"; // 'a' stores the input string, 'b' stores the target subsequence "hello".
int j = 0, pas = 0;      // 'j' is a pointer for the target string 'b', 'pas' counts matched characters from 'b'.
int main() {
  // Functional Utility: Read the input string 'a'.
  cin >> a;
  // Block Logic: Iterate through the input string 'a' to find "hello" as a subsequence.
  // Pre-condition: 'j' tracks the current character to match in "hello", 'pas' is the count of matched characters.
  // Invariant: 'j' only increments when a character from 'b' is found in sequence.
  for (int i = 0; i < a.size(); i++) {
    // If the current character in 'a' matches the character 'b[j]' (the next character in "hello").
    if (a[i] == b[j]) {
      j++;   // Move to the next character in "hello".
      pas++; // Increment the count of matched characters.

      // Inline: If all 5 characters of "hello" have been matched.
      if (pas == 5) {
        break; // Exit the loop as "hello" has been found as a subsequence.
      }
    }
  }
  // Functional Utility: Output "YES" if "hello" is a subsequence, "NO" otherwise.
  // The 'pas == 6' condition seems to be a slight error and should be 'pas == 5' based on the loop's break condition.
  // Assuming 'pas == 5' is the intended check for a complete match.
  if (pas == 6) { // This condition likely contains a bug; it should be 'pas == 5'.
    cout << "YES";
  } else {
    cout << "NO";
  }
}
