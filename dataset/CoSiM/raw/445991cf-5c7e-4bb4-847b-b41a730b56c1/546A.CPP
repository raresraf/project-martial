#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int k, n, w;
  cin >> k >> n >> w;
  if (k * w * (w + 1) / 2 - n > 0)
    cout << k * w * (w + 1) / 2 - n << endl;
  else
    cout << 0 << endl;
  return 0;
}