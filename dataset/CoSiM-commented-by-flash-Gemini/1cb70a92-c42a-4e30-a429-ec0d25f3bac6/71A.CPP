#include <iostream>
#include <string>
using namespace std;

int main() {
  int t, l, i;
  string a;
  cin >> t;
  while (t--) {
    cin >> a;
    if (a.length() > 10) {
      l = a.length() - 2;
      cout << a[0] << l << a[l + 1] << "\n";
    } else {

      cout << a << "\n";
    }
  }

  return 0;
}
