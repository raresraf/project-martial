/**
 * @file 58A.CPP
 * @brief Finds if "hello" is a subsequence in an input string.
 *
 * @details
 * This program solves the "Chat room" problem by checking if the word "hello"
 * can be formed by deleting zero or more characters from a given input string.
 * It uses a straightforward linear scan with two indices to track progress
 * through both the input string and the target word "hello".
 *
 * @section
 * Algorithm: Two-index linear scan for subsequence matching.
 * Time Complexity: O(N), where N is the length of the input string, as it requires a single pass.
 * Space Complexity: O(N) for storing the input string.
 */
#include <iostream>
using namespace std;

// Global variable for the input string. Using global variables is generally
// discouraged in larger projects but is common in small competitive programming scripts.
string a;
// Global variable for the target subsequence "hello".
string b = "hello";
// 'j' is a global index for the target string 'b'. It tracks which character of "hello" we are currently searching for.
int j = 0;
// 'pass' is a global counter for the number of characters successfully matched in the subsequence.
int pass = 0;

int main() {
  // Read the user's input string.
  cin >> a;

  /**
   * @brief This loop iterates through the input string 'a' to find the "hello" subsequence.
   *
   * Logic: The index 'i' scans through the input string 'a'. The index 'j' points
   * to the current character in the target string 'b' ("hello"). 'j' only
   * advances when a character from the subsequence is found in the correct order.
   */
  for (int i = 0; i < a.size(); i++) {
    // Pre-condition: Check if the current character in 'a' matches the character
    // we are looking for in 'b'.
    if (a[i] == b[j]) {
      // Invariant: A match is found, so we advance to look for the next character
      // in the subsequence.
      j++;
      pass++;
    }
  }

  // Post-condition: If 'pass' is 5, it means all five characters of "hello"
  // were found in the correct order.
  if (pass == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
