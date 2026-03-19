/**
 * @file raw/c6f71a93-0fb7-4e9f-a1cb-1af37843482b/281A.CPP
 * @brief Capitalizes the first letter of a given word.
 *
 * This program reads a single word from standard input and ensures that its
 * first character is uppercase. All other characters in the word remain
 * unchanged.
 *
 * @algorithm
 * 1. Read the input word into a string.
 * 2. Access the first character of the string at index 0.
 * 3. Use the `toupper()` standard library function to convert this character
 *    to its uppercase equivalent.
 * 4. Assign the resulting character back to the first position in the string.
 * 5. Print the modified string.
 *
 * @time_complexity O(L), where L is the length of the input string, dominated
 *                  by the I/O operations. The capitalization is O(1).
 * @space_complexity O(L) to store the input string.
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

// --- Competitive Programming Boilerplate ---
typedef vector<int> vi;
#define all(c) (c).begin(), (c).end()
#define pb push_back
// Debugging macro to print vector contents.
#define tr(a)                                                                  
  cout << "Vector: ";                                                          
  for (auto i : a)                                                             
    cout << i << " ";                                                          
  cout << "
";

int main() {
  // Fast I/O setup
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  string s;
  cin >> s;
  // The core logic: convert the first character to uppercase.
  s[0] = toupper(s[0]);
  cout << s;
  return 0;
}
