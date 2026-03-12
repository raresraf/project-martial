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
 * @57c92bbb-c149-4e32-81ed-704f73cf23a7/266A.CPP
 * @brief This program counts the number of adjacent identical characters in an input string.
 * This is typically used in competitive programming problems where a sequence of elements
 * (e.g., colored stones) needs analysis for consecutive identical occurrences.
 *
 * Algorithm: Single pass linear scan with a counter for adjacent matches.
 * Time Complexity: O(N) where N is the length of the input string.
 * Space Complexity: O(1) (excluding input string storage).
 */
#include <bits/stdc++.h> // Includes most standard C++ library headers.
using namespace std;     // Uses the standard namespace.

// Macro to optimize C++ standard I/O operations by decoupling C++ streams from C stdio
// and untie cin from cout, leading to faster input/output.
#define fastio                                                                 \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0)
#define LL long long          // Defines LL as a shorthand for long long.
#define mod 998244353         // Defines 'mod' for a common modulo value (e.g., in modular arithmetic problems).
#define FOR(i, j, k) for (int i = j; i < k; i++) // Macro for a common for-loop pattern.
#define ROF(i, j, k) for (int i = j; i >= k; i--) // Macro for a reverse for-loop pattern.

const long long INF = 1e18; // Defines INF as a large number, often used for infinity in algorithms.
const long long MAX = 1e5 + 10; // Defines MAX for a common maximum array/container size.

int main() {
  fastio; // Calls the fast I/O macro.
  int t = 1; // Initializes 't' to 1. In competitive programming, 't' often represents the number of test cases.
             // The commented line `cin>>t;` suggests it can be configured for multiple test cases.
  /**
   * Block Logic: Processes test cases.
   * Invariant: The code inside the loop is executed 't' times.
   */
  while (t--) { // Loop executes 't' times. For this snippet, it runs once.
    int n;       // Declares 'n' for the length of the string.
    string s;    // Declares a string variable to store the input sequence.
    cin >> n >> s; // Reads 'n' and the string 's' from standard input.
    int res = 0; // Initializes 'res' to 0, which will count adjacent identical characters.
    /**
     * Block Logic: Iterates through the string to count adjacent identical characters.
     * Starts from the second character (index 1) to compare with the previous one.
     * Invariant: 'res' accumulates the count of adjacent matches.
     */
    FOR(i, 1, n) // Loop from i = 1 to n-1 (inclusive).
      // Inline: Checks if the current character is the same as the preceding character.
      if (s[i] == s[i - 1]) 
        res++; // Increments 'res' if adjacent characters are identical.
    cout << res; // Prints the final count of adjacent identical characters.
  }
}
