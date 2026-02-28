/*              |\ | |  ||\ \ /(_~     |~)|_~|\/||_~|\/||~)|_~|~)
                |~\|_|/\||~\ | ,_)     |~\|__|  ||__|  ||_)|__|~\

                \ //~\| |    |\ |~)|_~    | ||\ ||/~\| ||_~
                | \_/\_/    |~\|~\|__    \_/| \||\_X\_/|__
                (J U S T   L I K E   E V E R Y O N E   E L S E !)

                             __ ,..---.._
                            +''''`--''-..`--..__
                           .\ _,/:i--._`:-:+._`.``-._
                          /`.._,,' \   `-.``--:.b....=.
                         |`..__,,..`.    '`.__::i--.-::_
                         )- .....--i'\..      --+`'''-'
                       ,' .'.._,.-'|._-b\
                      /,'<'    V   `oi| \             _.
                     |/ -|,--.." ,'-. ||\..      _.,;:'_<'
                     ''/ |  . ' |\||'\    /-'_/' `.
                    |,','|    , .    .-.|:.`.  + .,:..  |
                 ._,:'/ /-\   '^'   -Y"\ |.| || /,+8d| |
                .|/,'| |/':: ':=:' ,'|  | | \|| "+)='  |
                |+,';' /|_/ \     _/ \b':.\  \'| .||   ,'
                ,,:-i''_i' | ``-.Y',. ,|`: | \;- | |_,'
          __   |'| |i:'._ ,'     ,' ,; | |-)-'  __--:b__
         .P|   | |/,'|\  - ._   /  /   _,Y-   ,:/'  `.  `'".._
        ,'|| -','' | ._i._   `':| ,..,'     ,Y;'      \       `- ._
       |||||,..    | \ '-.._ _,' /       _,b-'         `.         '-.
       ||||P..i,  .| '....,-' _,'''''-'''               '    _,..    `\
       +'`   <'/  |`-.....---'                       ._         ,._
        |      |                                    ,'``,:-''''/,--`.
       Y|.b_,,:  |              ||                 ,;,Y'      /     |.
     ,' /'----' .'|   ..       |  |         '"   .`Y'     .,-b_....;;,.
    |+|,'     | | \.,  '      ,'  `:.  _     ,/__`     _=:  _,'``-
   / +,'      | /\_........:.'      '"----:::::'Y  .'.|   |||
   |' '      .'/- \                          /'|| || |   |||
   |||      /|     \L                        /'|| ||/ |   |||
   `.|    ,'/       .|                      / ,'||/o;/    |||
     `..._,,         |                      |/|   '       |||
       ``-'          |                      |,            |||
                     |          ,.          |             |||
  ,=--------....     |          ""          |             |||
,/,'.            i=..+._             ,..    '..;---:::''- | |
'/|           __....b `-''`---....../.,Y'''''j:.,.._      | `._
.'      _.Y.-'       `..       ii:,'--------' |     :-+. .| | b\
|     .=_,.---'''''--...:..--:'  /         _..-----..:=   | | '|\
|    '-''`'---                  ---'_,,,--''           `,.. |  | \.
 \  .                      ,' _,--''        :dg:      _,/ |||   |  \
`::b\`                 _,-i,-'                 ,..---'    ,|:|  | _|
`'--.:-._      ____,,,;.,'' `--._      ''''''''           |'|' .'  '
     ``'--....Y''-'              `''--..._..____._____...,' |  'o-'
                                             `''''`'''i==_+=_=i__
                                                     ||'''- '    `.
                                                      `-.......-''
*/
/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces.
 * @details The problem requires finding the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color.
 *
 * @algorithm
 * The solution iterates through the sequence of stones (represented as a string) and
 * counts the number of times two adjacent stones have the same color. This count
 * directly corresponds to the minimum number of stones that must be removed to
 * satisfy the condition.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of stones, as it requires a single pass
 * through the string.
 * Space Complexity: O(N) to store the string representing the colors of the stones.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @def fastio
 * @brief A macro to speed up C++ I/O operations.
 * It unties cin from cout and disables synchronization with C-style I/O.
 */
#define fastio                                                                 \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0)
#define LL long long
#define mod 998244353
#define FOR(i, j, k) for (int i = j; i < k; i++)
#define ROF(i, j, k) for (int i = j; i >= k; i--)

const long long INF = 1e18;
const long long MAX = 1e5 + 10;

int main() {
  // Optimizes I/O operations.
  fastio;
  int t = 1; // The problem has a single test case.
  while (t--) {
    int n;    // Variable to store the number of stones.
    string s; // Variable to store the colors of the stones.
    cin >> n >> s;
    int res = 0; // Counter for the minimum number of stones to remove.

    /**
     * @block
     * @brief Iterates through the stones to find adjacent pairs of the same color.
     * @invariant `res` holds the count of identical adjacent pairs found so far.
     * The loop starts from the second stone (index 1) to compare with its predecessor.
     */
    FOR(i, 1, n)
      // Pre-condition: Check if the current stone's color is the same as the previous one.
      if (s[i] == s[i - 1])
        res++; // Increment if an adjacent pair of same-colored stones is found.

    // Output the final count, which is the minimum number of removals.
    cout << res;
  }
}
