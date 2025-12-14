#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int k, n, w;
  cin >> k >> n >> w;
  cout << (((k * w * (w + 1) / 2) - n) >= 0 ? (k * w * (w + 1) / 2) - n : 0);
  return 0;
}