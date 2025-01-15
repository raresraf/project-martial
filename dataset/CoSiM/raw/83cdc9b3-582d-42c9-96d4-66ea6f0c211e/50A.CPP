#include <iostream>

using namespace std;

int main() {
  int n, k, c;
  cin >> n >> k;
  if ((n * k) % 2 > 0) {
    int a;
    a = (n * k) % 2;
    c = (n * k) - a;
  }
  c = (n * k) / 2;
  cout << c;

  return 0;
}