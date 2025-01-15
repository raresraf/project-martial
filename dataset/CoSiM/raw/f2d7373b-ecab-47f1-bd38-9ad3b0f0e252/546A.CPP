#include <iostream>
using namespace std;
int main() {
  int n, w, k, i, total = 0;
  cin >> k >> n >> w;
  for (i = 1; i <= w; i++) {
    total += i * k;
  }
  if ((total - n) <= 0)
    cout << "0\n";
  else
    cout << total - n << endl;
  return 0;
}
