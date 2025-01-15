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
  string s;
  cin >> s;
  int n = 0, a[100];
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+') {
      a[n] = (s[i] - '0');
      n++;
    }
  }
  sort(a, a + n);
  for (int i = 0; i < n - 1; i++) {
    cout << a[i] << "+";
  }
  cout << a[n - 1] << endl;
}
