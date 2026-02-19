/**
 * @file 58A.CPP
 * @brief Solution to a programming problem that checks if "hello" is a subsequence of a given string.
 * @details The program reads a string and determines if the characters of the word "hello" appear in the string in the correct order,
 * although not necessarily consecutively.
 *
 * Algorithm: Subsequence Matching
 * 1. Read the input string `str`.
 * 2. Define the target word, `word = "hello"`.
 * 3. Use a pointer `j` to track the current character in `word` (initially `j=0` for 'h').
 * 4. Iterate through the input string `str` from left to right.
 * 5. If the current character of `str` matches the character `word[j]`, it signifies a successful match for that part of the subsequence. Increment `j` to look for the next character of "hello".
 * 6. A counter `count` keeps track of how many characters from "hello" have been found in the correct order.
 * 7. If `count` reaches 5, the entire word "hello" has been formed as a subsequence. The loop can be terminated early.
 * 8. After the loop, if `count` is 5, print "YES". Otherwise, print "NO".
 *
 * Time Complexity: O(N), where N is the length of the input string, as it requires a single pass through the string.
 * Space Complexity: O(1), as the storage for the target word "hello" and a few counter variables is constant.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string str, word = "hello";
  int j = 0, count = 0;
  cin >> str;

  /**
   * Block Logic: Iterate through the input string to find the "hello" subsequence.
   * Pre-condition: 'str' is the input string, 'word' is "hello".
   * Invariant: 'j' points to the next character to be found in "hello", and 'count' is the number of characters already found.
   */
  for (int i = 0; i < str.length(); i++) {
    // If the current character in the input string matches the current character in the "hello" word.
    if (str[i] == word[j]) {
      j++;       // Move to the next character in "hello".
      count++;   // Increment the count of matched characters.
    }
    // Optimization: If all 5 characters of "hello" are found, no need to continue.
    if (count == 5)
      break;
  }
  
  /**
   * Block Logic: Output the result based on whether the full subsequence was found.
   */
  if (count == 5)
    cout << "YES";
  else
    cout << "NO";
}
