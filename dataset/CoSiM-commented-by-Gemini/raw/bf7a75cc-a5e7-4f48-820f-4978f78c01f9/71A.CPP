/**
 * @file 71A.CPP
 * @author Nikhil Nagrale
 * @brief Solution to Codeforces Problem 71A - "Way Too Long Words".
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 * @problem_link https://codeforces.com/problemset/problem/71/A
 * 
 * @details
 * This program shortens words that are "too long". A word is considered too long if its
 * length is strictly greater than 10. The abbreviation is formed by taking the first
 * character, followed by the number of characters between the first and last, and
 * finally the last character.
 *
 * @algorithm
 * The program reads the number of words, `n`. It then loops `n` times. In each
 * iteration, it reads a single word and checks its length.
 * - If the length is > 10, it constructs and prints the abbreviation.
 * - Otherwise, it prints the word as is.
 * This approach is space-efficient as it processes one word at a time without
 * storing the entire list.
 *
 * @complexity
 * - Time Complexity: O(L_total), where L_total is the sum of the lengths of all
 *   words, as processing is dominated by I/O.
 * - Space Complexity: O(L_max), where L_max is the length of the longest word,
 *   since only one word is stored in memory at a time.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  int n;
  cin >> n;
  
  /**
   * @brief Processes each of the `n` words.
   */
  while (n--) {
    string s;
    cin >> s;
    
    // Check if the word's length exceeds 10.
    if (s.length() > 10) {
      string ans;
      
      // Construct the abbreviation: first char + (length-2) + last char.
      ans.push_back(s.front());
      ans += to_string(s.length() - 2);
      ans += s.back();
      cout << ans << endl;

      // The author notes an alternative, more direct way to print the output:
      // cout<<s[0]<<s.size()-2<<s[s.size()-1]<<endl;
    } else {
      // If the word is not too long, print it without modification.
      cout << s << endl;
    }
  }
  return 0;
}
