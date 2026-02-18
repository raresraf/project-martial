/**
 * @file 339A.CPP
 * @brief Solution to the "Helpful Maths" problem (339A) from Codeforces.
 * @details The program takes a string representing a sum of numbers (1, 2, or 3),
 * such as "3+2+1", and rearranges it into non-decreasing order, e.g., "1+2+3".
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution involves parsing the string, sorting the numbers, and then reconstructing the sorted string.
 * 1. Read the entire sum as a single string.
 * 2. Create a vector of integers to hold the numbers from the sum.
 * 3. Iterate through the input string:
 *    - If a character is a digit ('1', '2', or '3'), convert it to an integer and add it to the vector.
 *    - If a character is a '+', increment a counter for the plus signs.
 * 4. Sort the vector of numbers in non-decreasing order.
 * 5. Reconstruct the output string by iterating through the sorted vector, printing each number,
 *    and interleaving them with '+' signs.
 *
 * @complexity
 * Time Complexity: O(L log L), where L is the length of the input string. The sorting
 * of the numbers is the dominant operation. The number of digits is proportional to L.
 * Space Complexity: O(L), to store the input string and the vector of numbers.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#define vc vector<int>
#define gcd(x, y) __gcd(x, y)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define MAX 1000001
#define inf 999999999
#define PI acos(-1)
#define MOD 1000000007

using namespace std;

int main() {
  string s;
  vc v;

  // The while loop is a general structure; for this problem, it will only execute once.
  while (cin >> s) {
    v.clear();
    int pls = 0; // Counter for the '+' operators.
    int len = s.size();
    /**
     * @block
     * @brief Parses the input string to extract numbers and count operators.
     * @pre 's' contains the input sum string.
     * @post 'v' contains all the numbers from the string, and 'pls' holds the count of '+'.
     */
    for (int i = 0; i < len; i++) {
      if (s[i] == '1')
        v.push_back(1);
      else if (s[i] == '2')
        v.push_back(2);
      else if (s[i] == '3')
        v.push_back(3);
      else
        pls++;
    }

    sort(v.begin(), v.end());

    int sz = v.size();

    /**
     * @block
     * @brief Reconstructs and prints the sorted sum.
     * @pre 'v' is the sorted vector of numbers, 'pls' is the count of operators.
     * @post The sorted sum string is printed to standard output.
     */
    for (int i = 0; i < sz; i++) {
      printf("%d", v[i]);
      if (pls > 0) {
        printf("+");
        pls--;
      }
    }
    printf("\n");
  }
  return 0;
}
