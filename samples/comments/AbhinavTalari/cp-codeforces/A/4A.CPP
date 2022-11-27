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

// Watermelon.cpp

int main() {

  int t;
  cin >> t;
  if (t == 1)
    cout << "NO" << endl;
  else if (t / 2 > 0 && t % 2 == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}