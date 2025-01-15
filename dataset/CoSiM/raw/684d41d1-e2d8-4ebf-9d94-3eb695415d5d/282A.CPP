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

//  282-A Bit++
int main() {

  int n;
  cin >> n;
  int x = 0;
  while (n--) {
    string op;
    cin >> op;

    if (op[0] == 'X') {
      if ((op.substr(1, 4)).compare("++") == 0)
        x++;

      if ((op.substr(1, 4)).compare("--") == 0)
        x--;

    }

    else if (op[op.length() - 1] == 'X') {
      if ((op.substr(0, 2)).compare("--") == 0)
        --x;

      if ((op.substr(0, 2)).compare("++") == 0)
        ++x;
    }
  }

  cout << x << endl;

  return 0;
}