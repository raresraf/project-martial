/**
 * @8cc97a0f-a199-4cef-b530-3fd55f9adae8/96A.CPP
 * @brief This program checks if a "dangerous situation" occurs in a football match.
 * A dangerous situation is defined as having seven consecutive players of the same team
 * (represented by '0' or '1' characters) in a sequence. The program reads a string
 * representing the player sequence and determines if this pattern exists.
 * Problem Link: https://codeforces.com/problemset/problem/96/A
 * Algorithm: Iterative string scanning for consecutive identical characters.
 * Time Complexity: O(L), where L is the length of the input string, due to a single linear pass.
 * Space Complexity: O(L) to store the input string.
 */

#include <bits/stdc++.h> // Includes various standard libraries, often used in competitive programming.

using namespace std;

int main() {
  // @var str: Stores the input string representing the sequence of players.
  string str;
  // @var flag: Counter for consecutive identical characters.
  // @var i: Loop index for iterating through the string.
  int flag = 0, i = 0;

  cin >> str; // Read the input player sequence string.
  
  // Block Logic: Iterate through the string to find seven consecutive identical characters.
  // Invariant: 'flag' counts consecutive identical characters.
  // Pre-condition: 'str' contains the player sequence.
  // Potential Issue: The loop condition `i <= str.size()` and access `str[i-1]` when `i` is 0
  // can lead to out-of-bounds access if 'str' is not empty and `i` starts at 0.
  // For safety, 'i' should ideally start from 1 when comparing with `str[i-1]`.
  while (i <= str.size()) { 
    // Block Logic: Check if the current character is the same as the previous one.
    // This comparison is problematic when i=0 as str[i-1] is out of bounds.
    if (str[i] == str[i - 1]) { 
      flag++; // Increment consecutive count.
      // Block Logic: If seven consecutive identical characters are found.
      if (flag == 6) { // Note: The problem asks for 7, so flag==6 means 7 identical characters.
        cout << "YES" << endl; // Output "YES" for a dangerous situation.
        return 0; // Terminate the program.
      }
    }
    // Block Logic: If the current character is different from the previous one.
    else {
      flag = 0; // Reset the consecutive count.
    }
    i++; // Move to the next character.
  }
  
  // Post-condition: If the loop completes without finding seven consecutive identical characters.
  cout << "NO" << endl; // Output "NO" if no dangerous situation is found.

  return 0; // Explicitly return 0 for successful execution.
}
