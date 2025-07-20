#include <iostream>

using namespace std;

int main() {
  string a;
  int n;
  cin >> n;
  while (n--) {
    cin >> a;
    if (a.length() > 10)
      cout << a[0] << (a.length() - 2) << a.back() << "\n";
    else
      cout << a << "\n";
  }
}
