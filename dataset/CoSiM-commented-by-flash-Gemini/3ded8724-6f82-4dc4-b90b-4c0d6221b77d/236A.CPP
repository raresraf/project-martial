/**
 * @file 236A.CPP
 * @brief Solution for competitive programming problem: "Boy or Girl".
 *
 * @details This program takes a username (string) and determines if the user
 *          is a "boy" or "girl" based on the number of unique characters in the username.
 *          If the count of unique characters is even, the output is "CHAT WITH HER!".
 *          If the count is odd, the output is "IGNORE HIM!".
 *          Algorithm: Character frequency counting using an unordered map to find unique characters.
 *          Time Complexity: O(L), where L is the length of the input string, as each
 *                           character is inserted into the hash map in average O(1) time.
 *          Domain: Competitive Programming.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s; // Declare string 's' to store the input username.
  cin >> s;  // Read the username string.
  
  // Declare an unordered_map to store character frequencies.
  // The keys are characters, and values are their counts.
  unordered_map<char, int> mp;
  
  // Block Logic: Iterate through the input string and count character frequencies.
  // Invariant: 'mp' will contain a count for each unique character in 's'.
  for (int i = 0; i < s.size(); i++) {
    mp[s[i]]++; // Increment the count for the current character.
  }
  
  // Get the number of unique characters (which is the size of the map).
  int n = mp.size();
  
  // Block Logic: Determine the output based on whether the number of unique characters is even or odd.
  if (n % 2 == 0) { // If the number of unique characters is even.
    cout << "CHAT WITH HER!" << endl;
  } else { // If the number of unique characters is odd.
    cout << "IGNORE HIM!" << endl;
  }
  
  return 0; // Indicate successful program termination.
}
