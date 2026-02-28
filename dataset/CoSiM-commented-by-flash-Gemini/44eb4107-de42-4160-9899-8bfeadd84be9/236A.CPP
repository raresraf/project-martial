/**
 * @file 236A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 236A - "Boy or Girl").
 *
 * This program determines a gender-like classification ("Boy" or "Girl") for a username
 * based on the parity of the number of unique characters it contains.
 * If the number of unique characters is even, the output is "CHAT WITH HER!";
 * if odd, the output is "IGNORE HIM!".
 *
 * Algorithm: Character frequency counting using a fixed-size array and parity check.
 * It uses an integer array (`a[26]`) as a frequency map for lowercase English alphabet
 * characters. It iterates through the input string, incrementing the count for each
 * character. Then, it iterates through the frequency array to count how many distinct
 * characters appeared. Finally, it checks the parity of this unique character count.
 *
 * Time Complexity: O(L + K), where L is the length of the input string and K is the
 * size of the alphabet (26). Since K is constant, this is effectively O(L).
 * Space Complexity: O(K) for the `a[26]` array. Since K is constant, this is effectively O(1).
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++.
#include <iostream>      // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads a username, counts its unique characters using a frequency array, and prints a message
 * based on the count's parity.
 */
int main() {
  string s; // Variable to store the input username.
  // Pre-condition: Read the input username string.
  cin >> s;
  int a[26] = {0}; // Declare and initialize a frequency array for 26 lowercase English letters.
                   // Index 0 for 'a', 1 for 'b', ..., 25 for 'z'.
  // Block Logic: Iterate through each character of the input string to build the frequency map.
  // Invariant: 'i' represents the current index in the string.
  for (int i = 0; i < s.length(); i++) {
    // Functional Utility: Convert character to 0-indexed alphabet position and increment its count.
    a[s[i] - 97]++; // ASCII value of 'a' is 97.
  }
  int res = 0; // Counter for the number of unique characters.
  // Block Logic: Iterate through the frequency array to count unique characters.
  // Invariant: 'i' represents the current alphabet index.
  for (int i = 0; i < 26; i++) {
    // If the count for a character is greater than 0, it means the character appeared at least once.
    if (a[i] > 0)
      res++; // Increment the unique character count.
  }
  // Block Logic: Check the parity of the unique character count.
  if (res % 2 == 0)      // If the number of unique characters is even.
    cout << "CHAT WITH HER!"; // Print message for "Girl".
  else                 // If the number of unique characters is odd.
    cout << "IGNORE HIM!"; // Print message for "Boy".
  cout << endl; // Add a newline for cleaner output.
  return 0; // Indicate successful execution.
}
