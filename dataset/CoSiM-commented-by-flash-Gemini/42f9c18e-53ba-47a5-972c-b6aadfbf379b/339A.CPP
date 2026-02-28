/**
 * @file 339A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 339A - "Helpful Maths").
 *
 * This program takes an arithmetic expression consisting of numbers 1, 2, and 3
 * separated by '+' signs, and outputs a new expression with the numbers sorted
 * in non-decreasing order.
 *
 * Algorithm: Parse, Sort, and Reconstruct.
 * It first parses the input string, extracting only the digit characters ('1', '2', '3')
 * and converting them to integers, which are then stored in a `std::vector`.
 * The vector is sorted using `std::sort`, and finally, the sorted numbers are
 * printed back to a string format, separated by '+' signs.
 *
 * Time Complexity: O(L + N log N), where L is the length of the input string,
 * and N is the number of digits (which is approximately L/2). This simplifies
 * to O(L log L) in the worst case due to the sorting step.
 * Space Complexity: O(L) for storing the input string and the `std::vector` of numbers.
 */

/***

            Bismillahir Rahmanir Rahim
            Read the name of Allah, who created you!!!
            Author : Shah Newaj Rabbi Shishir
            Department of CSE, City University, Bangladesh.

***/

#include <bits/stdc++.h> // Comprehensive header file for competitive programming.
using namespace std;

// Competitive programming macros for shorthand.
#define sf scanf
#define pf printf
#define scase sf("%d", &tc)
#define sn sf("%d", &n)
#define whilecase while (tc--)
#define eof while (cin >> n)
#define forloop for (pos = 1; pos <= tc; pos++)
#define arrayloop (i = 0; i < n; i++)
#define cinstr cin >> str
#define getstr getline(cin, str)
#define pcase pf("Case %d: ", pos)
#define pb push_back // Functional Utility: Appends an element to a vector.
#define in insert
#define llu unsigned long long
#define lld long long
#define u unsigned int

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads arithmetic expressions, extracts and sorts the numbers, then prints the sorted expression.
 * Designed to handle multiple test cases (implicit through `while(cin >> str)`).
 */
int main(void) {
  string str;        // Variable to store the input sum string.
  int len, i, val, siz; // 'len' for string length, 'i' for loop counter,
                       // 'val' for digit value, 'siz' for vector size.

  // Block Logic: Reads input strings continuously until End-Of-File or input stream fails.
  // Invariant: Each iteration processes one input arithmetic expression.
  while (cin >> str) {
    len = str.length(); // Get the length of the current input string.

    vector<int> v; // Vector to store the extracted integer numbers.

    // Block Logic: Iterate through the input string to extract only the digit characters.
    // Pre-condition: 'str' contains digits and '+' characters.
    // Invariant: 'i' advances, skipping '+' characters.
    for (i = 0; i < len; i++) {
      if (str[i] == '+')
        continue; // Skip '+' characters.

      val = str[i] - '0'; // Functional Utility: Convert character digit to integer.
      v.pb(val);          // Add the integer digit to the vector.
    }

    // Functional Utility: Sort the extracted numbers in non-decreasing order.
    sort(v.begin(), v.end());

    siz = v.size(); // Get the size of the vector.

    // Block Logic: Iterate through the sorted numbers to reconstruct and print the string.
    // Invariant: 'i' traverses the sorted vector 'v'.
    for (i = 0; i < siz; i++) {
      // If it's the last element, print it followed by a newline.
      if (i == siz - 1)
        cout << v[i] << endl;
      else
        // Otherwise, print the number followed by a '+'.
        cout << v[i] << "+";
    }
  }

  return 0;
}
