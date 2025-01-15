#include "bits/stdc++.h"
// PRAY :DOLPHINGARLIC: orz orz orz
#define GS                                                                     \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(nullptr);                                                            \
  cout.tie(0)
#define Int long long
#define all(x) x.begin(), x.end()
#define el '\n'
#define ints unsigned
#define Ints unsigned long long
#define elif else if
#define test                                                                   \
  int t;                                                                       \
  cin >> t;                                                                    \
  while (t--)

using namespace std;

int main() {
  GS;
  string a;
  cin >> a;
  int cnt{};
  int n = a.size();
  for (int i{}; i < n; i++) {
    if (a[i] - '0' == 7 || a[i] - '0' == 4)
      cnt++;
  }
  cout << (cnt == 7 || cnt == 4 ? "YES" : "NO") << endl;
  return 0;
}
/*
1:44 PM
9/4/2020
*/
