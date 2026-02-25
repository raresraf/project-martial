/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces (Problem 71A).
 *        This program takes 'n' words as input. For each word, if its length
 *        exceeds 10 characters, it's abbreviated. The abbreviation consists
 *        of the first character, followed by the number of characters between
 *        the first and last, and then the last character. Otherwise, the word
 *        is printed as is.
 * Algorithm: String Length Check and Abbreviation
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum
 *                  length of a word. String operations (length, push_back, to_string)
 *                  can take time proportional to string length.
 * Space Complexity: O(L) to store the longest word and the abbreviated string.
 */

/*
    Problem Link    :   https://codeforces.com/problemset/problem/71/A
    Contest Link    :   https://codeforces.com/contest/71/problem/A
    Status          :   Accepted

    Author          :   Nikhil Nagrale
    Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
    Codechef        :   https://www.codechef.com/users/nikhilnagrale2
    Github          :   https://github.com/nikhilnagrale2

    Copyright       :   Solve it on your own.
*/

#include <bits/stdc++.h> // Includes various standard libraries, e.g., iostream, string.
using namespace std;

/**
 * @brief Main function to process words and abbreviate them if they are too long.
 * @return 0 if the program executes successfully.
 */
int main() {
  int n; // Declare an integer to store the number of words.
  // Read the number of words from standard input.
  cin >> n;
  
  /**
   * Block Logic: Processes 'n' words, applying abbreviation logic to each.
   * Invariant: 'n' decrements after each word, until all words are processed.
   */
  while (n--) { // Loop 'n' times, once for each word.
    string s; // Declare a string to store the current word.
    // Read the current word from standard input.
    cin >> s;
    
    /**
     * Block Logic: Checks the length of the word and applies abbreviation if necessary.
     * Pre-condition: 's' contains the input word.
     * Post-condition: Either the abbreviated word or the original word is printed.
     */
    if (s.length() > 10) { // If the length of the word is greater than 10.
      string ans; // Declare a string to build the abbreviated answer.
      ans.push_back(s.front()); // Add the first character of the word.
      // Convert the length minus 2 to a string and append it.
      // s.length() - 2 represents the count of characters between the first and last.
      ans += to_string(s.length() - 2); 
      ans += s.back(); // Add the last character of the word.
      cout << ans << endl; // Print the abbreviated word.

      // Example of an alternative way to achieve the same:
      // cout<<s[0]<<s.size()-2<<s[s.size()-1]<<endl;
    } else
      cout << s << endl; // If the word is not too long, print it as is.
  }
  return 0; // Indicate successful program termination.
}