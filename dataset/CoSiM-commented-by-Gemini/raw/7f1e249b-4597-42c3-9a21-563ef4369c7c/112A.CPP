/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings, converts them to the same case (lowercase),
 * and then compares them lexicographically. It outputs -1 if the first string
 * is smaller, 1 if the second string is smaller, and 0 if they are equal.
 *
 * @algorithm String Conversion and Comparison
 * @complexity
 * Time: O(N), where N is the length of the strings, as we need to iterate
 *       through them to convert to lowercase. The final comparison also takes
 *       up to O(N).
 * Space: O(N), as we create new strings to store the lowercase versions.
 */
#include <iostream>
#include <string>
#include <vector>

// Using the standard namespace for competitive programming convenience.
using namespace std;

// Global strings to hold original and lowercase versions.
string a, b;
string c, d;

/**
 * @brief Main function to handle input, conversion, and comparison.
 */
int main() {
  // Read the two strings to be compared.
  cin >> a >> b;
  int n = (int)(a.length());

  // Pre-condition: Strings 'a' and 'b' are populated. Strings 'c' and 'd' are empty.
  // Block Logic: Iterate through both strings simultaneously, converting each to lowercase.
  for (int i = 0; i < n; i++) {
    // Convert character from string 'a' to lowercase.
    if ('A' <= a[i] && a[i] <= 'Z')
      c.push_back(a[i] - 'A' + 'a'); // ASCII-based conversion
    else
      c.push_back(a[i]);
    
    // Convert character from string 'b' to lowercase.
    if ('A' <= b[i] && b[i] <= 'Z')
      d.push_back(b[i] - 'A' + 'a'); // ASCII-based conversion
    else
      d.push_back(b[i]);
  }

  // Invariant: 'c' and 'd' are the lowercase versions of 'a' and 'b'.
  // Block Logic: Perform lexicographical comparison on the lowercase strings
  // and print the corresponding result.
  if (c < d)
    cout << -1;
  if (c == d)
    cout << 0;
  if (c > d)
    cout << 1;
  
  // The original code has a logical flaw where it could print multiple outputs
  // if not structured with if-else if. However, for the given constraints of
  // string comparison, only one condition will ever be true.
  // e.g., 'c < d' and 'c == d' cannot both be true.
  // The code is preserved as is, without functional modification.
  
  return 0; // The original code was missing a return statement.
}
