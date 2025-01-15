#include <bits/stdc++.h>
using namespace std;
#define MOD 100000007
#define ll long long

int main() {
  ll a, b;
  cin >> a >> b;
  cout << (a * b) % MOD;
  cout << endl;
  cout << ((a % MOD) * (b % MOD)) % MOD;
  return 0;
}
