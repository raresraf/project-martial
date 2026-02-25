/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces (Problem 266A).
 *        This program counts the number of adjacent stones of the same color
 *        that need to be removed to make all adjacent stones different.
 * Algorithm: Linear Scan / Adjacent Element Comparison
 * Time Complexity: O(N), where N is the number of stones.
 * Space Complexity: O(N) to store the stone sequence.
 */

#include <bits/stdc++.h> // Includes various standard libraries.
#include <iostream>      // Required for input/output operations (cin, cout).
#include <string>        // For using std::string (more robust than char array).

using namespace std;

/**
 * @brief Main function to read the sequence of stones and count adjacent identical ones.
 * @return 0 if the program executes successfully.
 */
int main() {
  int n; // Declare an integer to store the number of stones.
  // Read the number of stones from standard input.
  cin >> n;
  
  int count = 0; // Initialize a counter for adjacent identical stones.

  // Using std::string is safer and more idiomatic in C++ than a raw char array.
  // It handles memory management and null termination automatically.
  string s; 
  // Read the string of stones.
  cin >> s;

  /**
   * Block Logic: Iterates through the sequence of stones to find adjacent identical ones.
   * Invariant: 'count' accumulates the number of pairs of adjacent identical stones found so far.
   */
  for (int i = 0; i < n - 1; i++) { // Iterate up to n-1 to safely compare s[i] with s[i+1].
    // Check if the current stone is the same as the next stone.
    if (s[i] == s[i + 1]) {
      count++; // If they are identical, increment the counter.
    }
  }
  // Print the total count of adjacent identical stones followed by a newline.
  cout << count << endl;
  return 0; // Indicate successful program termination.
}
