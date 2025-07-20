#include <bits/stdc++.h>
#define ll long long int
#define loop(k) for (i = 0; i < k; ++i)
#define loop2(k, l) for (j = k; j < l; ++j)
#define mod 1000000007
using namespace std;
int main() {
  std::ios_base::sync_with_stdio(false); // cin.tie(NULL);
  ll n = 1, i, j;
  ll a[5][5];
  loop(5) loop2(0, 5) cin >> a[i][j];
  loop(5) {
    loop2(0, 5) {
      if (a[i][j] == 1)
        goto prin;
    }
  }
// cout<<i<<" "<<j<<"\n";
prin:
  int count = abs(2 - i) + abs(2 - j);
  ;
  cout << count;
  cout << "\n";
  return 0;
}