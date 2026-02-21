/**
 * @file 96A.CPP
 * @brief Determines if a situation in a football match is "dangerous".
 *
 * This program analyzes a string representing a sequence of football players
 * from two teams (encoded as '0' and '1'). A situation is considered
 * "dangerous" if there are seven or more consecutive players from the same team.
 *
 * Algorithm:
 * 1. Read the input string `s` representing the line of players.
 * 2. Initialize a counter `count` to 1.
 * 3. Iterate through the string from the second character to the end.
 * 4. In each iteration, compare the current character `s[i]` with the
 *    previous character `s[i-1]`.
 *    a. If they are the same, increment the `count` of consecutive players.
 *    b. If `count` reaches 7, a dangerous situation is found. Print "YES"
 *       and terminate the program.
 *    c. If the characters are different, reset `count` to 1.
 * 5. If the loop completes without `count` ever reaching 7, no dangerous
 *    situation occurred. Print "NO".
 *
 * Time Complexity: O(N), where N is the length of the string, as it involves
 * a single pass over the input.
 * Space Complexity: O(N) to store the input string.
 *
 * @note This file contains many preprocessor definitions, typedefs, and
 *       commented-out code blocks that are part of a common competitive
 *       programming template and are not used in this solution.
 */
#include <bits/stdc++.h>
// Unused competitive programming macros
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

// Unused template code for directional arrays
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
  // `count` tracks the number of consecutive identical players.
  int count = 1;
  int l = s.length();
  
  /**
   * @brief Iterate through the string to check for 7 consecutive players.
   * Pre-condition: `s` contains the player sequence.
   * Loop starts from 1 to allow comparison with the previous character.
   */
  for (int i = 1; i < l; ++i) {
    if (s[i] == s[i - 1]) {
      // If the current player is from the same team as the previous one, increment count.
      count += 1;
      // Check if the dangerous condition is met.
      if (count == 7) {
        cout << "YES" << endl;
        return 0; // Terminate immediately.
      }
    } else {
      // If the streak is broken, reset the counter.
      count = 1;
    }
  }

  // If the loop completes, no dangerous situation was found.
  cout << "NO" << endl;
  return 0;
}
