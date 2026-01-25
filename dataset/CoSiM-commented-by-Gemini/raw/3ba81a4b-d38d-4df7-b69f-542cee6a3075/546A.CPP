#include <iostream>
using namespace std;

#define end return 0

typedef long long ll;
typedef long long int lli;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(0);
  lli n, k, w, a, b, c;
  cin >> n >> k >> w;
  a = (w * (w + 1) / 2) * n;
  b = a - k;
  if (a - k < 0) {
    cout << 0 << endl;
  } else {
    cout << b << endl;
  }
  return 0;
}
