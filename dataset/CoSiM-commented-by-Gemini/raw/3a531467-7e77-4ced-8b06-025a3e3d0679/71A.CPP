/**
 * @file 71A.CPP
 * @brief A C-style C++ solution for the "Way Too Long Words" problem (Codeforces 71A).
 *
 * @details This program abbreviates words longer than 10 characters. It uses
 * C-style I/O (`scanf`/`printf`) and character arrays for string manipulation.
 * The numerous includes and macros are part of a generic competitive
 * programming template and are mostly unused for this specific problem.
 *
 * @algorithm The program reads `N`, the number of words to process. It then
 * enters a loop that runs `N` times. Inside the loop, it reads a word into a
 * char array, calculates its length, and checks if the length is greater than 10.
 * If it is, an abbreviated version is printed; otherwise, the original word is printed.
 *
 * @complexity
 * Time Complexity: O(N * L), where N is the number of words and L is their maximum
 *                  length, dominated by the I/O operations.
 * Space Complexity: O(L_max), for the fixed-size character buffer `msk`, which is
 *                   effectively O(1) given a constant maximum length.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// Unused macros from a competitive programming template.
#define fi first
#define se second
#define pb push_back
#define mp make_pair
#define pi 2 * acos(0.0)
#define eps 1e-9
#define PII pair<int, int>
#define PDD pair<double, double>
#define LL long long

using namespace std;

int N, len;
char msk[200]; // Buffer to hold the input word.

int main() {
  scanf("%d", &N);
  
  /**
   * @brief Block Logic: Processes each of the N words.
   */
  while (N--) {
    scanf("%s", msk);
    len = strlen(msk);
    if (len < 11)
      printf("%s\n", msk);
    else
      printf("%c%d%c\n", msk[0], len - 2, msk[len - 1]);
  }
  return 0;
}
