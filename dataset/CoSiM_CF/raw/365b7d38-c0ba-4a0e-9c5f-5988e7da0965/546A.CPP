#include <iostream>
using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int x = ((w * (w + 1)) / 2) * k;
  if (x <= n)
    cout << "0" << endl;
  else
    cout << x - n << endl;
}