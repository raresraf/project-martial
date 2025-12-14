#include <iostream>
using namespace std;
int main() {
  int k, n, w;
  cin >> k >> n >> w;
  cout << max(w * (w + 1) * k / 2 - n, 0);
}
