#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define vec vector<ll>
#define vec1 vector<vector ll> >
#define set set<ll>
#define mset multiset<ll>
#define pair pair<ll, ll>
#define pb push_back
void solve() {
  int n;
  cin >> n;
  int a[3] = {3, 5, 7};
  int c[3] = {0, 0, 0};
  if (n == 4 || n == 1 || n == 2)
    cout << -1 << endl;
  else if (n % 3 == 0)
    cout << n / 3 << " " << 0 << " " << 0 << endl;
  else if (n % 5 == 0)
    cout << 0 << " " << n / 5 << " " << 0 << endl;
  else if (n % 7 == 0)
    cout << 0 << " " << 0 << " " << n / 7 << endl;
  else {
    if (n > 7) {
      c[0] = n / 3;
      n = n % 3;
      if (n == 1)
        cout << c[0] - 2 << " " << 0 << " " << 1 << endl;
      else if (n == 2)
        cout << c[0] - 1 << " " << 1 << " " << 0 << endl;
    }
  }
}

int main() {
  int t;
  cin >> t;
  while (t--) {
    solve();
  }
  return 0;
}
