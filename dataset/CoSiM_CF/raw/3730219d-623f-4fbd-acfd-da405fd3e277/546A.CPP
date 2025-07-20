#include <iostream>

using namespace std;

int main() {

  int k, n, w;
  int tot = 0;

  cin >> k >> n >> w;

  for (int i = 1; i <= w; ++i) {
    tot += k * i;
  }

  if (tot > n) {
    cout << tot - n;
  }

  else {
    tot = 0;
    cout << tot;
  }

  return 0;
}
