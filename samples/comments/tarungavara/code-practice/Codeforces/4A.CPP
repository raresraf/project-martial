#include <iostream>

using namespace std;
int main() {
  int w;
  cin >> w;
  if (w >= 4) {
    if (w % 2 == 0)
      cout << "YES\n";
    else
      cout << "NO\n";
  } else
    cout << "NO\n";
  return 0;
}
