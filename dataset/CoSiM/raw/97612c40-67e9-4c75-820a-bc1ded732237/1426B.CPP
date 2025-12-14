/*
This code is written by Shammi Anand
contact : shammianand101@gmail.com, shammianand.me
*/
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define pb push_back
#define f first
#define s second
#define vi vector<int>
#define pii pair<int, int>
#define rep(i, a, n) for (int i = a; i < n; i++)
#define F(i, n) for (int i = 0; i < n; i++)
#define all(a) a.begin(), a.end()
#define INF 1e9 + 7
#define nl "\n"
#define w(x)                                                                   \
  int x;                                                                       \
  cin >> x;                                                                    \
  while (x--)

template <typename T_vector>
void output(const T_vector &v, int start = -1, int end = -1) {
  if (start < 0)
    start = 0;
  if (end < 0)
    end = int(v.size());

  for (int i = start; i < end; i++) {
    cout << v[i] << (i < end - 1 ? ' ' : '\n');
  }
}

void shammi() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
}

int main() {
  shammi();
  w(t) {
    int n, m;
    cin >> n >> m;
    int matrices[n][2][2];
    for (int i = 0; i < n; i++) {
      cin >> matrices[i][0][0] >> matrices[i][0][1] >> matrices[i][1][0] >>
          matrices[i][1][1];
    }
    if (m % 2) {
      cout << "NO\n";
    } else {
      bool ok = false;
      for (int i = 0; i < n; i++) {
        if (matrices[i][0][1] == matrices[i][1][0]) {
          ok = true;
          break;
        }
      }
      cout << (ok ? "YES\n" : "NO\n");
    }
  }
  return 0;
}