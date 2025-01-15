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

  string s1, s2;
  cin >> s1 >> s2;
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  int x = s1.compare(s2);
  if (x < 0)
    cout << -1 << endl;
  if (x > 0)
    cout << 1 << endl;
  if (x == 0)
    cout << 0 << endl;

  return 0;
}