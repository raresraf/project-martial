#include <bits/stdc++.h>
using namespace std;
#define Test(x)                                                                \
  int x;                                                                       \
  cin >> x
#define sz(x) ((int)x.size())
#define sqr(x) ((x) * (x))
#define pb push_back
#define mp make_pair

int main() {
  // ios_base::sync_with_stdio(0);
  Test(w);
  if (w != 2 && w % 2 == 0)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
