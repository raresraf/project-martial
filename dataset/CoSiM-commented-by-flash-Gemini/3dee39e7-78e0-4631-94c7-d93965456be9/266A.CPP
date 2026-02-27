/**
 * @file 266A.CPP
 * @brief Solution for competitive programming problem: "Stones on the Table".
 *
 * @details This program calculates the minimum number of stones to remove from a
 *          row so that no two adjacent stones have the same color. This is achieved
 *          by simply counting the number of adjacent pairs of identical stones.
 *          Algorithm: Single pass linear scan.
 *          Time Complexity: O(N), where N is the number of stones, as each stone
 *                           is examined once.
 *          Domain: Competitive Programming.
 */
#include <cctype>
#include <iostream>
using namespace std;
/*
    Author- Sohanuzzaman Soad
    University of Asia Pacific (UAP),BD

*/
int main() {
  // Declare 'n' for the number of stones.
  int n;
  // Read the total number of stones from standard input.
  cin >> n;
  // Declare string 'a' to store the sequence of stone colors.
  string a;
  // Read the string representing stone colors.
  cin >> a;
  
  // Initialize counters for adjacent identical stones of each color.
  int red = 0;   // Count of adjacent 'R' stones to be removed.
  int green = 0; // Count of adjacent 'G' stones to be removed.
  int blue = 0;  // Count of adjacent 'B' stones to be removed.
  
  // Block Logic: Iterate through the stones, comparing each stone with its predecessor.
  // Invariant: 'red', 'green', 'blue' accumulate counts of adjacent identical stones.
  for (int i = 1; i < n; i++) { // Start from the second stone (index 1) to compare with the previous.
    // Precondition: If the current stone is the same color as the previous one.
    if (a[i - 1] == a[i])
      // Block Logic: Increment the specific color counter based on the stone's color.
      if (a[i] == 'R') {
        ++red;   // Increment red stone removal count.
      } else if (a[i] == 'G') {
        ++green; // Increment green stone removal count.
      } else if (a[i] == 'B') {
        ++blue;  // Increment blue stone removal count.
      }
  }
  // Output the total minimum number of stones to remove (sum of all adjacent identical stones).
  cout << red + green + blue;
  
  return 0; // Indicate successful program termination.
}
