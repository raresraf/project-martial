#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  int n;
  cin >> n;

  int c = 0, max = 0, l, e;
  while (n--) {
    cin >> l >> e;
    c -= l;
    c += e;
    if (c > max)
      max = c;
  }
  cout << max;
  return 0;
}
