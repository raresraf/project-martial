#include <iostream>
using namespace std;

int main() {
  int i, j, k, n, cont = 0;
  cin >> n;
  while (n--) {
    cin >> i >> j >> k;
    if (i + j + k > 1)
      cont++;
  }
  cout << cont << endl;
  return 0;
}
