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

// 231ATeam
int main() {

  int n;
  cin >> n;
  int fc = 0;

  for (int i = 0; i < n; i++) {
    int ic = 0;
    int x;
    for (int j = 0; j < 3; j++) {
      cin >> x;
      if (x == 1)
        ic++;
    }
    if (ic >= 2)
      fc++;
  }
  cout << fc << endl;

  return 0;
}