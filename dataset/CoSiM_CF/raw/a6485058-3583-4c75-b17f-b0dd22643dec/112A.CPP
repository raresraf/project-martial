#include <cstring>
#include <iostream>

using namespace std;

int main() {
  string a, b;

  cin >> a >> b;

  int k = 2;

  for (int i = 0; i < a.length(); i++) {
    if (a[i] >= 97) {
      a[i] = a[i] - 32;
    }

    if (b[i] >= 97) {
      b[i] = b[i] - 32;
    }

    if (a[i] < b[i]) {
      k = -1;
      break;
    } else if (a[i] > b[i]) {
      k = 1;

      break;
    } else {
      k = 0;
    }
  }

  cout << k << endl;

  return 0;
}
