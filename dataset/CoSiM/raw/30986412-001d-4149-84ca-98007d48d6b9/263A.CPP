#include <algorithm>
#include <cmath>
#include <iostream>
using namespace std;

#define end return 0

typedef long long ll;
typedef unsigned long long int ulli;
typedef long long int lli;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(0);
  lli a, x, y, c = 0;
  for (int i = 1; i < 6; ++i) {
    for (int j = 1; j < 6; ++j) {
      cin >> a;
      if (a == 1) {
        x = i;
        y = j;
      }
    }
  }
  c += abs(3 - x);
  c += abs(3 - y);
  cout << c << endl;

  return 0;
}
