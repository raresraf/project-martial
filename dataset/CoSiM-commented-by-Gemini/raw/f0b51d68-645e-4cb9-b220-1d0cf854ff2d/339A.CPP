/**
 * @file 339A.CPP
 * @brief Rearranges a sum of numbers (1, 2, 3) into non-decreasing order.
 *
 * This program takes a string representing a sum of the digits '1', '2', and '3'
 * (e.g., "3+1+2"), extracts the numbers, sorts them, and prints the rearranged sum.
 *
 * @b Algorithm:
 * 1. The input string is parsed to extract the digit characters.
 * 2. The characters are converted to integers and stored in a vector.
 * 3. The vector of integers is sorted in non-decreasing order.
 * 4. The sorted numbers are printed, separated by '+'.
 *
 * @b Time_Complexity: O(N log N), where N is the number of summands, dominated
 *    by the sorting algorithm. The string traversal is O(L), where L is the
 *    length of the string.
 * @b Space_Complexity: O(N) to store the summands in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  vector<int> v;
  string s;
  cin >> s;
  
  /**
   * @brief This loop extracts the digits from the input string.
   * It filters for characters '1' through '3' and converts them to integers.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] >= '1' && s[i] <= '3')
      // 's[i] - '0'' converts a digit character to its integer equivalent.
      v.push_back(s[i] - '0');
  }

  // Handle the edge case of a single number with no '+' signs.
  if (s.size() == 1)
    cout << s << endl;
  else {
    // Sort the vector of numbers.
    sort(v.begin(), v.end());
    
    // Print the sorted sum.
    for (int i = 0; i < v.size() - 1; i++) {
      cout << v[i] << "+";
    }
    cout << v[v.size() - 1] << endl;
  }
  return 0;
}
