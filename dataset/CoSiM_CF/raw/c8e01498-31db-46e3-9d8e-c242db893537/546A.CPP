#include <iostream>
using namespace std;
int main() {
  int a = 0, n = 0, cm = 0, bm = 0, nm = 0;
  cin >> a >> cm >> n;
  nm = (a * n * (n + 1)) / 2;
  bm = nm - cm;
  if (bm > 0)
    cout << bm << endl;
  else
    cout << "0" << endl;

  return 0;
}
