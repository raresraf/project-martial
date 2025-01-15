#include <iostream>

using namespace std;

int main() {
  int k, n, w, l;
  l = 0;
  cin >> k;
  cin >> n;
  cin >> w;

  for (int i = 1; i <= w; i++) {
    l = l + i * k;
  }

  if ((l - n) <= 0)
    cout << "0";
  else
    cout << l - n;

  return 0;
}
