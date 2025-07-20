#include <bits/stdc++.h>
#include <iostream>
using namespace std;
typedef long long int ll;
typedef long double ld;
#define mp make_pair
#define v vector
#define pb push_back
#define fi first
#define se second
#define ALL(v) v.begin(), v.end()
#define UN(v)                                                                  \
  sort((v).begin(), (v).end()), v.erase(unique(v.begin(), v.end()), v.end())
#define N 31622779
#define mod 998244353

// 50-A Dominos
int main() {

  int n, k;
  cin >> n >> k;

  int x = ((n * k) / 2) > 0 ? (n * k) / 2 : 0;
  cout << x << endl;

  return 0;
}