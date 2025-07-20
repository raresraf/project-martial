#include <iostream>
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);
using namespace std;
int main() {
  optimizar_io string a, res = "";
  cin >> a;
  for (int i = 0; i < (int)a.size(); ++i) {
    a[i] = tolower(a[i]);
    if (a[i] != 'a' && a[i] != 'e' && a[i] != 'i' && a[i] != 'o' &&
        a[i] != 'u' && a[i] != 'y')
      res = res + "." + a[i];
  }
  cout << res << "\n";
  return 0;
}
