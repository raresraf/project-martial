/**
 * @file 58A.CPP
 * @brief Determines if the word "hello" is a subsequence of a given string.
 *
 * This program reads a string and checks if it's possible to obtain the word
 * "hello" by deleting zero or more characters from the input string.
 *
 * @algorithm Greedy two-pointer subsequence check.
 * @complexity
 * Time: O(N), where N is the length of the input string, because we iterate
 *       through the input string exactly once.
 * Space: O(1) auxiliary space, besides the storage for the input string.
 */
#include <iostream>
#include <string>
#include <cstdio> // For printf

// Using the standard namespace for competitive programming convenience.
using namespace std;

/**
 * @brief Main function to read the input and perform the subsequence check.
 */
int main() {
  // s: the input string provided by the user.
  // r: the target subsequence, "hello".
  string s, r = "hello";
  
  // i: pointer for the input string 's'.
  // j: pointer for the target string 'r'. It tracks which character of "hello" we are currently looking for.
  int i, j = 0;
  
  // Read the input string.
  cin >> s;
  
  // Pre-condition: 'j' is 0, pointing to the first character of "hello".
  // Block Logic: Iterate through the input string 's' to find characters of "hello" in order.
  for (i = 0; i < s.length(); i++) {
    // If we've already found all characters of "hello", we can stop early.
    if (j == 5) break; 
    
    // If the current character in 's' matches the character we're looking for in 'r',
    // advance the pointer 'j' to look for the next character of "hello".
    if (s[i] == r[j])
      j++;
  }
  
  // Invariant: After the loop, 'j' holds the number of characters from "hello"
  // that were found in the correct order.
  // Block Logic: If 'j' is 5, it means all five characters of "hello" were found.
  if (j == 5)
    printf("YES");
  else
    printf("NO");
}
