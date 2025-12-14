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

  int index_i, index_j;
  vector<vector<int>> matrix(5, vector<int>(5, 0));
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> matrix[i][j];
      if (matrix[i][j] == 1) {
        index_i = i;
        index_j = j;
      }
    }
  }

  int dx = 2, dy = 2;
  int ddx = abs(index_i - dx), ddy = abs(index_j - dy);
  cout << ddx + ddy;
  return 0;
}