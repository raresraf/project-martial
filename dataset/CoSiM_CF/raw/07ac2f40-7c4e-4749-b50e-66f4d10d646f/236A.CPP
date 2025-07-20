#include <bits/stdc++.h>
using namespace std;

#define rep(i, a, b) for (ll i = a; i < b; i++)
#define re(i, b) for (ll i = 0; i < b; i++)
#define repr(i, n) for (ll i = n - 1; i >= 0; i--)
#define ll long long
#define ld long double
#define llu long long unsigned
#define pll std::pair<ll, ll>
#define ppll std::pair<ll, pll>
#define vll std::vector<ll>
#define vpll std::vector<pll>
#define vppll std::vector<ppll>
#define mll std::unordered_map<ll, ll>
#define mpll std::unordered_map<pll, ll>
#define mppll std::unordered_map<ppll, ll>
#define sll std::set<ll>
#define ff first
#define ss second
#define msll std::multiset<ll>
#define all(c) c.begin(), c.end()
#define allr(c) c.rbegin(), c.rend()
#define srt(x) sort(all(x))
#define rsrt(x) sort(allr(x))
#define eb emplace_back
#define pb push_back
#define s(yy)                                                                  \
  ll yy;                                                                       \
  cin >> yy
#define mod 1000000007
#define maxlong 1e18 + 5

/*####################################################
#                                                    #
#                     @author                        #
#                  Parth Lathiya                     #
#      https://www.cse.iitb.ac.in/~parthiitb/        #
#                                                    #
####################################################*/

int main() {

  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

#ifdef PARTH_LATHIYA_HOME

  freopen("236_A.txt", "r", stdin);

  ll ttt, bkkk;
  cin >> ttt;
  bkkk = ttt;
  while (ttt--) {
    cout << "Testcase - " << bkkk - ttt << "\n";
#endif

    //--------------------------------------------------------------------------------------

    string s;

    cin >> s;
    unordered_map<char, ll> m;
    re(i, s.length()) m[s[i]]++;
    if (m.size() & 1)
      cout << "IGNORE HIM!";
    else
      cout << "CHAT WITH HER!";
      //--------------------------------------------------------------------------------------

#ifdef PARTH_LATHIYA_HOME
    cout << "\n";
  }
#endif

  return 0;
}