/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem from Codeforces (Problem 236A).
 * This program determines if a given username (string) corresponds to a "boy" or "girl"
 * based on the parity of the number of distinct characters in the username.
 *
 * Algorithm: Count distinct characters. Sort the string to group identical characters,
 * then iterate to count unique ones. Determine gender based on parity (odd for girl, even for boy).
 * Time Complexity: O(L log L) due to sorting, where L is the length of the username string.
 * Space Complexity: O(L) for storing the string.
 */

#include <bits/stdc++.h> // @brief Includes many standard libraries (e.g., iostream, string, algorithm).
using namespace std;
int main() {
  /**
   * @brief Main function to solve the Boy or Girl problem.
   * Reads a username string, counts its distinct characters, and determines
   * if the count is odd ("CHAT WITH HER!") or even ("IGNORE HIM!").
   */
  string s; // @brief The input username string.
  cin >> s;
  
  // Block Logic: Sort the string to bring identical characters together.
  // This simplifies the process of counting distinct characters.
  sort(s.begin(), s.end());
  
  int l = s.length(); // @brief Length of the string.
  int ans = 0; // @brief Counter for distinct characters.
  
  // Block Logic: Count the number of distinct characters in the sorted string.
  // Iterate from the second character, comparing with the previous one.
  // If they are different, it indicates a new distinct character.
  for (int i = 1; i < l; i++) {
    if (s[i] != s[i - 1])
      ans++; // Inline: Increment distinct character count.
  }
  
  // Note: The commented-out lines below are for debugging and not part of the core logic.
  //    cout<<s<<endl;
  //    cout<<ans<<endl;
  
  // Block Logic: Determine the output based on the parity of the distinct character count.
  // If 'ans' (number of distinct characters) is odd, it's a "girl".
  if (ans & 1) // Inline: Bitwise AND with 1 checks for oddness.
    cout << "CHAT WITH HER!" << endl;
  // Otherwise, it's a "boy".
  else
    cout << "IGNORE HIM!" << endl;

  return 0; // @brief Indicates successful program termination.
}
