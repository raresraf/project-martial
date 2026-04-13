/**
 * @file 58A.CPP
 * @brief This program determines if the word "hello" can be formed as a subsequence
 * from a given input string. A subsequence means the characters appear in the correct
 * order, but not necessarily consecutively.
 * Algorithm: State-machine based character matching. It uses a counter ('cnt') to track
 * the progress of finding each character of "hello" in sequential order.
 * Time Complexity: O(N), where N is the length of the input string, due to a single pass.
 * Space Complexity: O(N) for storing the input string, plus O(1) for the counter.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience.
using namespace std;

int main() {
  string str; // Stores the input string.
  int cnt = 0; // Acts as a state counter for finding characters of "hello" (0=initial, 1='h' found, ..., 5='o' found).
  int l;     // Stores the length of the input string.
  // Reads the input string from standard input.
  cin >> str;
  // Gets the length of the input string.
  l = str.size();
  /**
   * Block Logic: Iterates through the input string 'str' to find the characters of "hello" in order.
   * The 'cnt' variable manages the state of the search:
   *   - cnt = 0: Looking for 'h'.
   *   - cnt = 1: Looking for 'e' (after 'h' is found).
   *   - cnt = 2: Looking for first 'l' (after 'e' is found).
   *   - cnt = 3: Looking for second 'l' (after first 'l' is found).
   *   - cnt = 4: Looking for 'o' (after second 'l' is found).
   * Invariant: `cnt` accurately represents the number of characters of "hello" found in correct order as a subsequence.
   */
  for (int i = 0; i < l; i++) {
    // If 'h' is found and it's the first character we're looking for.
    if (cnt == 0 && str[i] == 'h') {
      cnt++; // Transition to state 1 (h found).
      // cout<<cnt<<str[i]<<endl; // Commented-out debugging line.
    }
    // Else if 'e' is found and we're looking for it (after 'h').
    else if (cnt == 1 && str[i] == 'e') {
      cnt++; // Transition to state 2 (e found).
      // cout<<cnt<<str[i]<<endl; // Commented-out debugging line.
    }
    // Else if 'l' is found and we're looking for one of the 'l's (after 'e').
    else if (cnt >= 2 && cnt <= 3 && str[i] == 'l') {
      cnt++; // Transition to state 3 or 4 (first or second 'l' found).
      // cout<<cnt<<str[i]<<endl; // Commented-out debugging line.
    }
    // Else if 'o' is found and we're looking for it (after both 'l's).
    else if (cnt == 4 && str[i] == 'o') {
      cnt++; // Transition to state 5 (o found).
      // cout<<cnt<<str[i]<<endl; // Commented-out debugging line.
    }
    // else { // No operation if character doesn't match current state's expectation.
    // }
  }
  /**
   * Conditional Logic: After processing the entire string, determines if "hello" was successfully formed.
   * If 'cnt' reached 5, it means all characters of "hello" were found in the correct order as a subsequence.
   */
  if (cnt == 5) {
    cout << "YES"; // Prints "YES" if "hello" is a subsequence.
  } else {
    cout << "NO"; // Prints "NO" otherwise.
  }
  // Standard program exit.
  return 0;
}
