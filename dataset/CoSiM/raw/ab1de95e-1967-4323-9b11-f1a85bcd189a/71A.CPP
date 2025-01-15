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

// 71-A Way too long words
int main() {

  int t;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    if (s.length() > 10)
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    else
      cout << s << endl;
  }

  return 0;
}