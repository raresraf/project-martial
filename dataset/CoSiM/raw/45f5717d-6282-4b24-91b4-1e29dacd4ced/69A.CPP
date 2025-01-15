#include <bits/stdc++.h>
#define ll long long int
#define loop(k) for (i = 0; i < k; ++i)
#define loop2(k, l) for (j = k; j < l; ++j)
#define mod 1000000007
using namespace std;
int main() {
  std::ios_base::sync_with_stdio(false); // cin.tie(NULL);
  ll n = 1, i, j;
  cin >> n;
  ll a[n][3];
  loop(n) loop2(0, 3) cin >> a[i][j];
  int sum1 = 0, sum2 = 0, sum3 = 0;
  loop(n) {
    sum1 += a[i][0];
    sum2 += a[i][1];
    sum3 += a[i][2];
  }
  if (sum1 || sum2 || sum3) {
    cout << "NO";
  } else
    cout << "YES";
  cout << "\n";
  return 0;
}