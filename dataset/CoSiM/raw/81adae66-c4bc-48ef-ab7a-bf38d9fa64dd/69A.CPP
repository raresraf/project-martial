#include <iostream>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;
typedef double dbl;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(false);
  ulli n;
  cin >> n;
  lli a, b, c, d = 0, e = 0, f = 0;
  while (n--) {
    cin >> a >> b >> c;
    d += a;
    e += b;
    f += c;
  }
  if (!d && !e && !f) {
    cout << "YES" << endl;
  } else
    cout << "NO" << endl;
  return 0;
}