#include <iostream>
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);
using namespace std;
int main() {
  optimizar_io long long a, x, y, z, cont = 0, con = 0, conti = 0;
  for (cin >> a; a--;) {
    cin >> x >> y >> z;
    cont += x;
    con += y;
    conti += z;
  }
  if (conti == 0 && con == 0 && conti == 0)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
