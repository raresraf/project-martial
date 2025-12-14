#include <iostream>
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);
using namespace std;
int main() {
  optimizar_io long long cont = 0, conti = 0;
  string a, b;
  cin >> a >> b;
  for (int i = 0; i < (int)a.size(); ++i) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
    cont += (char)a[i];
    conti += (char)b[i];
    if (conti > cont) {
      cout << "-1\n";
      return 0;
    }
    if (conti < cont) {
      cout << "1\n";
      return 0;
    }
  }
  cout << "0\n";
  return 0;
}
