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

typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;

const int MOD = 1e9 + 7;
const int INF = 1e9 + 5;
const ll LINF = LLONG_MAX;
int main() {
  int t;
  cin >> t;
  int c2 = 0;
  while (t--) {
    int a, b, c;
    int c1 = 0;
    cin >> a >> b >> c;
    if (a == 1)
      c1++;
    if (b == 1)
      c1++;
    if (c == 1)
      c1++;
    if (c1 >= 2)
      c2++;
  }
  cout << c2 << endl;
}