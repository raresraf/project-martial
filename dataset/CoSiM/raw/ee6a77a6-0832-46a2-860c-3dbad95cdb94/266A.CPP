#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define X first
#define Y second
#define PB push_back
#define F0(I, N) for (ll I = 0; I < N; I++)
#define F1(I, N) for (ll I = 1; I <= N; I++)
#define F(I, X, N) for (ll I = X; I < N; I++)
#define R0(I, N) for (ll I = N - 1; I >= 0; I--)
#define R1(I, N) for (ll I = N; I > 0; I--)
#define R(I, X, N) for (ll I = N - 1; I >= X; I--)
#define A(X) X.begin(), X.end()

ll n, t;
string s;
void solve() {
  cin >> n >> t;
  cin.ignore();
  cin >> s;
  F0(j, t) {
    F(i, 1, n) {
      if (s[i] == 'G' && s[i - 1] == 'B') {
        s[i] = 'B';
        s[i - 1] = 'G';
        i++;
      }
    }
  }
  cout << s << "\n";
}
int main() {
  cin.tie(NULL);
  ios_base::sync_with_stdio(false);
  solve();
  return 0;
}
