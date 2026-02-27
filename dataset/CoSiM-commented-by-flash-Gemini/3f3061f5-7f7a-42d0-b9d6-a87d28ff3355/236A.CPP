/**
 * @file 236A.CPP
 * @brief Solution for competitive programming problem: "Boy or Girl".
 *
 * @details This program takes a username (string) and determines if the user
 *          is a "boy" or "girl" based on the number of unique characters in the username.
 *          If the count of unique characters is odd, the output is "IGNORE HIM!".
 *          If the count is even, the output is "CHAT WITH HER!".
 *          Algorithm: Iterates through all possible lowercase English letters ('a' to 'z')
 *                     and for each, checks its presence in the input string to count
 *                     the total number of unique characters.
 *          Time Complexity: O(A * L), where A is the size of the alphabet (26 for English)
 *                           and L is the length of the input string.
 *          Domain: Competitive Programming.
 */
#include <iostream>
using namespace std;
int main() {
  string s; // Declare string 's' to store the input username.
  // Declare 'flag' as a boolean indicator (1 if character found, 0 otherwise)
  // and 'c' as a counter for unique characters.
  int flag, c = 0;
  cin >> s;  // Read the username string.

  // Block Logic: Iterate through all lowercase English letters ('a' to 'z').
  // Invariant: 'c' accumulates the count of unique lowercase letters present in 's'.
  for (int i = 'a'; i <= 'z'; i++) { // 'i' takes on ASCII values of 'a' through 'z'.
    flag = 0; // Reset flag for each new character being checked.
    // Block Logic: Check if the current letter 'i' is present in the input string 's'.
    // Invariant: 'flag' is set to 1 if 'i' is found in 's'.
    for (int j = 0; j < s.length(); j++)
      if (s[j] == i) // If the character 'i' is found in 's'.
        flag = 1;   // Set flag to true.
    
    if (flag == 1) // If the character 'i' was found in the string 's'.
      c++;         // Increment the unique character count.
  }
  
  // Block Logic: Determine the output based on whether the count of unique characters is odd or even.
  if (c % 2) // If 'c' is odd.
    cout << "IGNORE HIM!";
  else // If 'c' is even.
    cout << "CHAT WITH HER!";
}
