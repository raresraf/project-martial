#include <iostream>

using namespace std;

int main() {
  int n;
  int maxCap = 0;
  int currentCap = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;
    currentCap += b;
    currentCap -= a;
    if (currentCap >= maxCap) {
      maxCap = currentCap;
    }
  }
  cout << maxCap << endl;
  return 0;
}