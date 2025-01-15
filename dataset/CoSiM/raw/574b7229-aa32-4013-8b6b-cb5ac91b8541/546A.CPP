#include <iostream>
using namespace std;
int main() {
  int i, k, w, n, s = 0;
  cin >> k >> n >> w;
  for (i = 1; i <= w; i++) {
    s += i * k;
  }
  if (s > n)
    cout << s - n << endl;
  else
    cout << "0" << endl;
}
