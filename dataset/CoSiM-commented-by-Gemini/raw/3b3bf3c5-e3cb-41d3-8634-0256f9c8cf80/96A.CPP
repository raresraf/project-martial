///       From dust I came and to dust I shall return
/**
                    Asadullah Bin Rahman
                          CSE-17
    HAJEE MOHAMMAD DANESH SCIENCE AND TECHNOLOGY UNIVERSITY
*/
/**
 * @file 96A.CPP
 * @brief Solution for the "Football" problem (Codeforces 96A).
 *
 * @details The problem is to determine if there is a "dangerous" situation, defined
 * as a sequence of 7 or more players of the same team standing next to
 * each other. Players are represented by a string of '0's and '1's.
 *
 * @algorithm The program uses a single pass with a counter to find the longest
 * run of identical consecutive characters.
 * 1. It initializes a `count` for the current run to 1.
 * 2. It iterates through the string from the second character.
 * 3. If the current character matches the previous one, `count` is incremented.
 *    If not, the run is broken, and `count` is reset to 1.
 * 4. If `count` ever reaches 7, a dangerous situation is found, "YES" is printed,
 *    and the program terminates.
 * 5. If the loop finishes without `count` reaching 7, "NO" is printed.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string, due to the
 *                  single pass.
 * Space Complexity: O(L), to store the input string.
 */
#include <bits/stdc++.h>
// Unused macros and template code from a competitive programming setup.
#define sf(f) scanf("%d", &f)
#define pf(f) printf("%d\n", f)
#define PI 3.1415926535897932385
#define size 3000000
#define mp make_pair
#define pb push_back
#define one(x) __builtin_popcount(x)     /// int
#define onel(x) __builtin_popcountl(x)   /// long int
#define onell(x) __builtin_popcountll(x) /// long long
using namespace std;
typedef long long ll;

/// int dx[]= {-1,-1,0,0,1,1};
/// int dy[]= {-1,0,-1,1,0,1};
/// int dx[]= {0,0,1,-1};                 /*4 side move*/
/// int dy[]= {-1,1,0,0};                 /*4 side move*/
/// int dx[]= {1,1,0,-1,-1,-1,0,1};       /*8 side move*/
/// int dy[]= {0,1,1,1,0,-1,-1,-1};       /*8 side move*/
/// int dx[]={1,1,2,2,-1,-1,-2,-2};       /*knight move*/
/// int dy[]={2,-2,1,-1,2,-2,1,-1};       /*knight move*/
/// Always Remember   'A'=65,'Z'=90 'a'=97 'z'=122 '0'=48

int main() {
  string s;
  cin >> s;
  int count = 1; // Tracks the length of the current run of identical characters.
  int l = s.length();

  /**
   * @brief Block Logic: Iterates through the string to find a run of 7.
   * @invariant `count` holds the length of the identical character run ending
   * at index `i-1`.
   */
  for (int i = 1; i < l; ++i) {
    if (s[i] == s[i - 1]) {
      // If the character is the same as the previous one, extend the run.
      // cout<<"fuck\n";
      count += 1;
      if (count == 7) {
        cout << "YES" << endl;
        return 0; // Early exit upon finding a dangerous situation.
      }
    }

    else
      // If the character changes, reset the run counter.
      count = 1;
  }

  // If the loop completes, no run of 7 was found.
  cout << "NO" << endl;
  return 0;
}
