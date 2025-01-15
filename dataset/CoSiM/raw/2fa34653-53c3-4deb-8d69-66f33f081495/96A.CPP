#include <bits/stdc++.h>
using namespace std;

#define faster ios_base::sync_with_stdio(false), cin.tie(NULL), cout.tie(NULL)
#define mp make_pair
#define mod 1000000007
#define qmod 998244353
#define endl "\n"
#define pb push_back
#define ff first
#define ss second
#define mk make_pair

typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<string> vs;

const int MOD = 1e9 + 7;
const int INF = 1e9 + 5;
const ll LINF = LLONG_MAX;
const int MAX_V = 201;
int main() {
  string s;
  cin >> s;
  int c = 1, i, j;
  for (i = 0; s[i] != '\0'; i++) {
    if (s[i] == s[i - 1]) {
      c += 1;
      if (c >= 7) {
        cout << "YES";
        return 0;
      }
    } else {
      c = 1;
    }
  }
  cout << "NO";
  return 0;
}
