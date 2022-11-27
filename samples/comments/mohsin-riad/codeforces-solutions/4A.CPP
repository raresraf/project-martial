#include <bits/stdc++.h>
using namespace std;

#define max 1000
typedef long long ll;
typedef double dd;

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  ll n;
  cin >> n;

  if (n == 2)
    cout << "NO";
  else if (n % 2 != 0)
    cout << "NO";
  else
    cout << "YES";
  return (0);
}
