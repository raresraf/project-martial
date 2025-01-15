/**
* Git >> Mohsin_Riad


















**/
#include <bits/stdc++.h>
using namespace std;

#define IOS                                                                    \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define mx 1000
#define endl "\n"
#define pb push_back
#define pob pop_back
typedef long long ll;
typedef double dd;
const int N = 1e6;

int main() {
  IOS

      ll n;
  cin >> n;

  ll x[n + 5], y[n + 5], z[n + 5];
  ll s1 = 0, s2 = 0, s3 = 0;
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
    s1 += x[i];
    s2 += y[i];
    s3 += z[i];
  }

  if (s1 == 0 && s2 == 0 && s3 == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return (0);
}
