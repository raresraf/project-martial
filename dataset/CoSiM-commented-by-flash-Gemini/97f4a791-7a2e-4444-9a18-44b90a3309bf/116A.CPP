#include <iostream>
using namespace std;

int main() {
  int n, currentPass = 0, maxCap = 0;
  cin >> n;

  while (n--) {
    int a, b;
    cin >> a >> b;

    currentPass += b - a;
    maxCap = max(maxCap, currentPass);
  }

  cout << maxCap << endl;
  return 0;
}
