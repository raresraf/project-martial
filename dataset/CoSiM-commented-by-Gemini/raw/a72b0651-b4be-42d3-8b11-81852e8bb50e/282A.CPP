/**
 * @file 282A.CPP
 * @brief Solution for CodeForces Problem 282A - "Bit++".
 * 
 * @details
 * This program simulates a series of operations on a variable `X`, which is initialized to 0.
 * The operations are given as strings: "++X", "X++", "--X", or "X--".
 * The program reads `n` such operations, stores them, and then processes them to find the final value of `X`.
 *
 * @algorithm
 * 1. Read the number of operations, `n`.
 * 2. Read all `n` operation strings and store them in a `std::vector`.
 * 3. Initialize a variable `X` to 0.
 * 4. Iterate through the vector of stored operation strings:
 *    a. For each string, use `string::find("++")` to check for an increment operation.
 *    b. If "++" is found, increment `X`.
 *    c. Otherwise, assume it's a decrement operation ("--") and decrement `X`.
 * 5. Print the final value of `X`.
 *
 * @time_complexity O(N*L), where N is the number of operations and L is the length of the operation string.
 *                  This is because `string::find` can take O(L) time.
 * @space_complexity O(N*L) to store all the operation strings in a vector. This is less memory-efficient
 *                   than processing the operations one by one.
 *
 * @note The file includes many headers and macros that are unused in this specific solution.
 */
#include <iostream>
#include <string>
#include <vector>

// Define common macros for brevity, though they can sometimes harm readability.
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define REV(i, b, a) for (int i = (a); i >= (b); i--)
#define mp make_pair
#define pb push_back

using namespace std;

int main(int argc, char *argv[]) {
  int n; // Number of operations.
  int X = 0; // The variable to be manipulated, initialized to 0.
  string s;
  vector<string> input;
  
  cin >> n;

  /**
   * @brief Read all operation strings and store them in a vector.
   */
  FOR(i, 0, n) {
    cin >> s;
    input.pb(s);
  }

  /**
   * @brief Process the stored operations.
   * Invariant: `X` holds the accumulated value from the operations processed so far.
   */
  FOR(i, 0, n) {
    // Check if the operation string contains "++".
    if (input.at(i).find("++") != std::string::npos) {
      X++;
    } 
    // Otherwise, assume it contains "--".
    else if (input.at(i).find("--") != std::string::npos) {
      X--;
    }
  }

  cout << X;

  return 0;
}
