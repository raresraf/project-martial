/**
 * @89a1ad02-b158-43d2-bc32-7bf26d5d24ce/58A.CPP
 * @brief This program determines if the string "hello" is a subsequence of a given input string.
 * It iterates through the input string, attempting to match characters of "hello" in order.
 * Algorithm: Subsequence check (greedy approach).
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(N) to store the input string.
 */

#include <iostream>
using namespace std;

// @var a: Stores the input string from the user.
// @var b: Stores the target subsequence string "hello".
// @var j: Pointer for iterating through the target subsequence string 'b'.
// @var pass: Counter for the number of matched characters from the target subsequence.
string a, b = "hello";
int j = 0, pass = 0;

int main() {
  // Block Logic: Read the input string.
  cin >> a;
  
  // Block Logic: Iterate through the input string to find "hello" as a subsequence.
  // Invariant: 'j' tracks the current character being sought from 'b'.
  // Pre-condition: 'a' contains the full input string, 'b' is "hello".
  for (int i = 0; i < a.size(); i++) {
    // Block Logic: If the current character in 'a' matches the character being sought in 'b'.
    if (a[i] == b[j]) {
      j++;    // Move to the next character in 'b'.
      pass++; // Increment match counter.
    }
  }
  
  // Block Logic: Determine if "hello" was successfully found as a subsequence.
  // Post-condition: Output "YES" if all characters of "hello" were matched, "NO" otherwise.
  if (pass == 5) { // 'hello' has 5 characters.
    cout << "YES";
  } else {
    cout << "NO";
  }
}
