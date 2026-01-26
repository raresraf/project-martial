#include <bits/stdc++.h> // This wont work only for g++ compiler.

#define for0(i, n) for (int i = 0; i < (int)(n); ++i)  // 0 based indexing
#define for1(i, n) for (int i = 1; i <= (int)(n); ++i) // 1 based indexing
#define forc(i, l, r)                                                          \
  for (int i = (int)(l); i <= (int)(r);                                        \
       ++i) // closed interver from l to r r inclusive
#define forr0(i, n) for (int i = (int)(n)-1; i >= 0; --i) // reverse 0 based.
#define forr1(i, n) for (int i = (int)(n); i >= 1; --i)   // reverse 1 based

// short hand for usual tokens
#define pb push_back
#define fi first
#define se second

// to be used with algorithms that processes a container Eg: find(aint(c),42)
#define aint(x) (x).begin(), (x).end()  // Forward traversal
#define raint(x) (x).rbegin, (x).rend() // reverse traversal

// traversal function to avoid long template definition. Now with C++11 auto
// ainteviates the pain.
#define tr(c, i)                                                               \
  for (__typeof__((c)).begin() i = (c).begin(); i != (c).end(); i++)

// find if a given value is present in a container. Container version. Runs in
// log(n) for set and map
#define present(c, x) ((c).find(x) != (c).end())

// find version works for aint containers. This is present in std namespace.
#define cpresent(c, x) (find(aint(c), x) != (c).end())

// Avoiding wrap around of size()-1 where size is a unsigned int.
#define sz(a) int((a).size())

#define FAST_INP                                                               \
  ios::sync_with_stdio(false);                                                 \
  cin.tie(0);
const unsigned int M = 1000000007;

using namespace std;

// Shorthand for commonly used types
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef long long int ll;
typedef vector<int> vint;
typedef vector<vint> vvint;
typedef double ld;

int main() {
  FAST_INP;
  // int T;
  // cin>>T;
  // while(T--){

  ll n;
  cin >> n;
  vector<ll> coins(n);
  for (ll i = 0; i < n; i++)
    cin >> coins[i];

  sort(coins.begin(), coins.end(), greater<int>());
  auto sum = accumulate(coins.begin(), coins.end(), 0);
  auto partial = sum / 2;

  auto check = 0, count = 0;
  for (ll i = 0; i < n; i++) {
    check += coins[i];
    count++;
    if (check > partial)
      break;
  }
  cout << count << "\n";
  //}
  return 0;
}