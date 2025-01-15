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
  int n;
  cin >> n;
  int ans = 0;
  int as = n * 2;
  int min{};
  for (int i = 0; i < as; i++) {
    int b{};
    cin >> b;
    int a{};
    cin >> a;
    min -= b;
    min += a;
    if (ans < min) {
      ans = min;
    }
  }
  cout << ans << endl;
  return 0;
}
/*
1:25 OM
9/4/2020
*/
