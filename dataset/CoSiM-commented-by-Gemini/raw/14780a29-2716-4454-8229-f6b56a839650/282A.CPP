/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem from Codeforces.
 * @details This program simulates a series of operations on a variable, initialized to zero.
 * The operations are given as strings containing either "++" for increment or "--" for decrement.
 * The program computes and prints the final value of the variable after all operations.
 *
 * @section algorithm Algorithm
 * 1. Initialize a counter variable to zero.
 * 2. Read the total number of operations, `n`.
 * 3. Loop `n` times:
 *    a. Read the operation string.
 *    b. Use `string::find()` to check for the presence of the "++" substring.
 *       If it's found, increment the counter.
 *    c. If "++" is not found, check for the "--" substring. If found, decrement the counter.
 * 4. After the loop completes, print the final value of the counter.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of operations. Although `string::find` is used, the string length is constant and small, making each check effectively O(1).
 * - Space Complexity: O(1), as the space required for the operation string is constant.
 */
#include <iostream>
using namespace std;

// A. Bit++
// https://codeforces.com/problemset/problem/282/A
int main() {
  int n, count = 0;
  cin >> n;
  string x;
  // This loop iterates through all the operations.
  for (int i = 0; i < n; i++) {
    cin >> x;
    // The operation is identified by searching for the "++" or "--" substring
    // within the input string `x`.
    if (x.find("++") != string::npos)
      count++; // Increment if "++" is found.
    else if (x.find("--") != string::npos)
      count--; // Decrement if "--" is found.
  }
  // Print the final result.
  cout << count;
  return 0;
}
