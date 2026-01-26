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

//  112-A Petya String
int main() {

  int a[5][5];
  pair<int, int> x;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1)
        x = make_pair(i + 1, j + 1);
    }
  }
  int fc = abs(3 - x.first) + abs(3 - x.second);
  cout << fc << endl;

  return 0;
}